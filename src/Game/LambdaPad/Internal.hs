{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}
module Game.LambdaPad.Internal where

import Control.Applicative ( Applicative, (<$>) )
import Control.Concurrent
  ( ThreadId, forkIO, yield )
import Control.Concurrent.MVar
  ( MVar, newMVar, isEmptyMVar, takeMVar, putMVar )
import Control.Monad.Reader.Class ( MonadReader, ask, local )
import Control.Monad.State.Strict
    ( State, StateT, evalStateT, execStateT, runState, runStateT )
import Control.Monad.State.Class ( MonadState, get, put )
import Control.Monad.Trans ( MonadIO, liftIO )
import Data.Int ( Int16 )
import Data.Monoid ( Monoid, mempty, mappend, mconcat )
import Data.Word ( Word8 )

import Data.Algebra.Boolean ( Boolean(..) )
import Control.Lens
    ( ALens', (.=), (%=), (%%=), (^.), cloneLens, to, use, view )
import Prelude hiding ( (&&), (||), not )
import Control.Lens.TH ( makeLenses )

import qualified Data.HashMap.Strict as HM
import qualified Data.Vector as V
import qualified SDL

-- Utilities

whenJust :: Monad m => Maybe a -> (a -> m ()) -> m ()
whenJust = flip $ maybe $ return ()

(.:) :: (c -> d) -> (a -> b -> c) -> a -> b -> d
(.:) = (.).(.)
infixr 1 .:

-- Types, instances and lenses

newtype Stop = Stop {runStop :: IO ()}

instance Monoid Stop where
  mempty = Stop $ return ()
  mappend stop1 stop2 = Stop $ stop stop1 >> stop stop2

stop :: Stop -> IO ()
stop = runStop

data ButtonState = Pressed | Released
  deriving (Show, Eq)

data Button = Button
    { _buttonState :: !ButtonState
    , buttonHash :: !Int
    }
makeLenses ''Button

instance Show Button where 
  show but = case but^.buttonState of
      Pressed -> "but(*)"
      Released -> "but( )"

data Direction = C | N | NE | E | SE | S | SW | W | NW
  deriving (Show, Eq)

data DPad = DPad
    { _dir :: Direction
    , dpadHash :: !Int
    }
makeLenses ''DPad

instance Show DPad where
    showsPrec _ dpad' =
        ("dpad["++) . (dpad'^.dir.to shows) .
        ("]"++)

data Trigger = Trigger
  { _pull :: !Float
  , triggerHash :: !Int
  }
makeLenses ''Trigger

instance Show Trigger where
  showsPrec _ trig = ("trig("++) . (trig^.pull.to shows) . (')':)

data Stick = Stick
    { _horiz :: !Float
    , _vert :: !Float
    , stickHash :: !Int
    }
makeLenses ''Stick

instance Show Stick where
  showsPrec _ stick = ("stick("++) . (stick^.horiz.to shows) . (","++) .
      (stick^.horiz.to shows) . (')':)

-- | This is the tilt of the stick measured by the clockwise distance around the
-- stick you are starting at up or N.  E.g. N = 0, S = 1/2, W = 1/4, and NE =
-- 7/8.
--
-- A neutral stick has tilt Nothing, but this should not be used to determine
-- whether it has any tilt due to noise.  Check instead whether the push is
-- greater than some small number, e.g. 0.2.
tilt :: Stick -> Maybe Float
tilt stick | x == 0 && y == 0 = Nothing
          | y == 0 = Just $ if x > 0 then 1/4 else 3/4
          | x == 0 = Just $ if y > 0 then 0 else 1/2
          | theta > 0 = Just $ if y > 0 then theta else 0.5 + theta
          | otherwise = Just $ if y > 0 then 1 + theta else 0.5 + theta
  where
    x = stick^.horiz
    y = stick^.vert
    -- Note that x and y are switched.  This equivalent to a flip across x = y.
    -- Yes, y could be 0, but laziness.
    theta = (atan $ x / y) / (2*pi)

-- | This is the amount an stick is displaced from center, where @0.0@ is neutral
-- and @1.0@ is fully displaced.
push :: Stick -> Float
push stick = sqrt $ (stick^.horiz.to sq) + (stick^.vert.to sq)
    where sq x = min 1.0 $ x*x

data Pad = Pad
    { _a :: !Button
    , _b :: !Button
    , _x :: !Button
    , _y :: !Button
    , _lb :: !Button
    , _rb :: !Button
    , _rs :: !Button
    , _ls :: !Button
    , _start :: !Button
    , _back :: !Button
    , _home :: !Button
    , _dpad :: !DPad
    , _leftTrigger :: !Trigger
    , _rightTrigger :: !Trigger
    , _leftStick :: !Stick
    , _rightStick :: !Stick
    }
  deriving (Show)
makeLenses ''Pad

-- A Convenience type for referring to 'Button's on the 'Pad'
type PadButton = ALens' Pad Button
-- A Convenience type for referring to the 'DPad' on the 'Pad'
type PadDPad = ALens' Pad DPad
-- A Convenience type for referring to 'Trigger's on the 'Pad'
type PadTrigger = ALens' Pad Trigger
-- A Convenience type for referring to 'Stick'es on the 'Pad'
type PadStick = ALens' Pad Stick

newtype Filter user = Filter { runFilter :: LambdaPadData user -> Bool }

makeFilterOp :: (Bool -> Bool -> Bool)
             -> Filter user -> Filter user -> Filter user
makeFilterOp op left right = Filter $ \tf ->
    runFilter left  tf `op` runFilter right tf

instance Boolean (Filter user) where
  true = Filter $ const True
  false = Filter $ const False
  not = Filter . (not.) . runFilter
  (&&) = makeFilterOp (&&)
  (||) = makeFilterOp (||)
  xor = makeFilterOp (/=)
  (<-->) = makeFilterOp (==)

type LambdaPadInner user = StateT (LambdaPadData user) IO

type LambdaPadMvar user = StateT (MVar (LambdaPadData user)) IO

newtype LambdaPad user a = LambdaPad { runLambdaPad :: LambdaPadInner user a }
  deriving (Monad, MonadIO, Functor, Applicative)

newtype PadState a = PadState
    { runPadState :: State Pad a }
  deriving (Monad, Functor, Applicative)

newtype GameWriter user a = GameWriter
    { runGameWriter :: LambdaPadInner user a }
  deriving (Monad, Functor, Applicative)

data PadConfig = PadConfig
    { padName :: String
    , buttonConfig :: Word8 -> ButtonState -> PadState (Maybe Button)
    , dpadConfig :: Word8 -> Word8 -> PadState (Maybe DPad)
    , axisConfig :: Word8 -> Int16 -> PadState (Maybe (Either Stick Trigger))
    }

data GameConfig user = GameConfig
    { gameName :: String
    , newUserData :: IO user
    , onStop :: user -> IO ()
    , onEvents :: GameWriter user ()
    }

data LambdaPadData user = LambdaPadData
    { _lpUserData :: !user
    , _lpJoystick :: SDL.Joystick
    , _lpPad :: !Pad
    , _lpOnTick :: LambdaPad user ()
    , _lpSpeed :: Float -- ^ In seconds.
    , _lpPadConfig :: !PadConfig
    , _lpEventFilter :: HM.HashMap Int [(Filter user, LambdaPad user ())]
    }
makeLenses ''LambdaPadData

instance MonadState user (LambdaPad user) where
  get = LambdaPad $ use lpUserData
  put = LambdaPad . (lpUserData.=)

instance MonadReader Pad (LambdaPad user) where
  ask = LambdaPad $ use lpPad
  local f m = LambdaPad $ get >>=
      (liftIO . evalStateT (lpPad %= f >> runLambdaPad m))

-- | This returns whether the provided 'Filter' matches the current game
-- state.  Note this works with any 'Filter', so you can use it with 'whenUser'
-- 'Filter's, too.
isPad :: Filter user -> LambdaPad user Bool
isPad = LambdaPad . flip fmap get . runFilter

-- | Gets current tick speed in ticks per second.
getSpeed :: LambdaPad user Float
getSpeed = LambdaPad $ use lpSpeed

-- | Since 'getSpeed' is typically so great
withResidual
    :: Float -- ^ The dead zone.
    -> Float -- ^ The speed in units per second.
    -> ALens' user Float -- ^ The lens to the residual.
    -> ALens' Pad Float -- ^ The lens to the axis.
    -> LambdaPad user Int
withResidual deadZone unitSpeed residual axis = do
    displacement <- view $ cloneLens axis.to sqSign
    tickSpeed <- LambdaPad $ use lpSpeed
    if abs displacement < deadZone
      then return 0
      else cloneLens residual %%=
          (splitIntFrac.(+displacement * unitSpeed / tickSpeed))
  where splitIntFrac val = (intVal, val - fromIntegral intVal)
          where intVal = truncate val :: Int
        sqSign x' = signum x'*x'*x'

neutralPad :: Pad
neutralPad = Pad
    { _a = Button Released 0
    , _b = Button Released 1
    , _x = Button Released 2
    , _y = Button Released 3
    , _lb = Button Released 4
    , _rb = Button Released 5
    , _rs = Button Released 6
    , _ls = Button Released 7
    , _start = Button Released 8
    , _back = Button Released 9
    , _home = Button Released 10
    , _dpad = DPad C 11
    , _leftTrigger = Trigger 0.0 12
    , _rightTrigger = Trigger 0.0 13
    , _leftStick = Stick 0.0 0.0 14
    , _rightStick = Stick 0.0 0.0 15
    }

-- Pad Config

instance MonadState Pad PadState where
  get = PadState $ get
  put = PadState . put

runLambdaPadState :: PadState a -> LambdaPadInner user a
runLambdaPadState padState = do
    (val, newPad) <- (runState $ runPadState padState) <$> use lpPad
    lpPad .= newPad
    return val

simpleButtonConfig
    :: [(Word8, PadButton)] -> Word8 -> ButtonState -> PadState (Maybe Button)
simpleButtonConfig rawMapping button state = do
    case HM.lookup button mapping of
      Nothing -> return Nothing
      Just but -> do
        (cloneLens but.buttonState) .= state
        fmap Just $ use $ cloneLens but
  where !mapping = HM.fromList rawMapping 

simpleDPadConfig ::
  Word8 -> [(Word8, Direction)] -> Word8 -> Word8 -> PadState (Maybe DPad)
simpleDPadConfig hatIndex rawMapping hat dirWord =
    if hatIndex == hat
      then do
        case HM.lookup dirWord mapping of
          Nothing -> return Nothing
          Just dir' -> do
            (dpad.dir) .= dir'
            Just <$> use dpad
      else return Nothing
  where !mapping = HM.fromList rawMapping 

simpleAxisConfig
    :: [(Word8, Int16 -> PadState (Either Stick Trigger))]
    -> Word8 -> Int16 -> PadState (Maybe (Either Stick Trigger))
simpleAxisConfig rawMapping axis val =
    case HM.lookup axis mapping of
      Nothing -> return Nothing
      Just axisAct -> Just <$> axisAct val
  where !mapping = HM.fromList rawMapping 

horizStickConfig :: PadStick -> Int16 -> PadState (Either Stick Trigger)
horizStickConfig stick rawVal = do
    (cloneLens stick.horiz) .= val
    fmap Left $ use $ cloneLens stick
  where
    val = fromIntegral (if rawVal == minBound then minBound + 1 else rawVal) /
              fromIntegral (maxBound :: Int16)

vertStickConfig :: PadStick -> Int16 -> PadState (Either Stick Trigger)
vertStickConfig stick rawVal = do
    (cloneLens stick.vert) .= val
    fmap Left $ use $ cloneLens stick
  where
    val = negate $ fromIntegral
        (1+if rawVal == maxBound then maxBound - 1 else rawVal) /
        fromIntegral (maxBound :: Int16)

triggerConfig :: PadTrigger -> Int16 -> PadState (Either Stick Trigger)
triggerConfig trig rawVal = do
    (cloneLens trig.pull) .= val
    fmap Right $ use $ cloneLens trig
  where
    val = (fromIntegral rawVal - fromIntegral (minBound :: Int16)) /
        (fromIntegral (maxBound :: Int16) - fromIntegral (minBound :: Int16))

-- Game config

class WithFilter input a where
  with :: ALens' Pad input -> a -> Filter user

whenPad :: (Pad -> Bool) -> Filter user
whenPad = Filter . flip (^.) . (lpPad.) . to

whenUser :: (user -> Bool) -> Filter user
whenUser userPred = Filter (^.lpUserData.to userPred)

instance WithFilter Button ButtonState where
  with but state = whenPad (^.cloneLens but.buttonState.to (==state))

instance WithFilter DPad Direction where
  with dpad' dir' = whenPad (^.cloneLens dpad'.dir.to (==dir'))

newtype Pull = Pull (Float -> Bool)

instance WithFilter Trigger Pull where
  with trig (Pull pullPred) = whenPad (^.cloneLens trig.pull.to pullPred)

data StickFilter = Horiz (Float -> Bool)
                | Vert (Float -> Bool)
                | Push (Float -> Bool)
                | Tilt (Float, Float)

stickWith :: PadStick -> (Stick -> Float) -> (Float -> Bool) -> Filter user
stickWith aLens getFloat stickPred = whenPad
    (^.cloneLens aLens.to getFloat.to stickPred)

instance WithFilter Stick StickFilter where
  with stick (Horiz horizPred) = stickWith stick (^.horiz)  horizPred
  with stick (Vert vertPred) = stickWith stick (^.vert) vertPred
  with stick (Push pushPred) = stickWith stick (^.to push) pushPred
  with stick (Tilt (at, range)) =
      whenPad (^.cloneLens stick.to tilt.to (maybe False inBounds))
    where
      fracPart val = abs $ val - fromIntegral (truncate val :: Int)
      lowerBound = fracPart $ at - (range / 2)
      upperBound = fracPart $ at + (range / 2)
      inBounds tilt'
          | lowerBound < upperBound =
              lowerBound <= tilt' && tilt' < upperBound
          | otherwise =
              upperBound < tilt' && tilt' <= lowerBound

instance WithFilter Stick Direction where
  with stick dir' = case dir' of
      C -> with stick $ Push (<0.2)
      N -> tiltAt 0
      NE -> tiltAt $ 1/8 
      E -> tiltAt $ 1/4
      SE -> tiltAt $ 3/8
      S -> tiltAt $ 1/2
      SW -> tiltAt $ 5/8
      W -> tiltAt $ 3/4
      NW -> tiltAt $ 7/8
    where
      tiltAt at =
          with stick (Push (>=0.25)) && with stick (Tilt (at, 1/8))

onHash :: (a -> Int) -> ALens' Pad a -> Filter user -> LambdaPad user ()
       -> GameWriter user ()
onHash aHash aLens filter' act = GameWriter $ do
    hashVal <- use $ lpPad.cloneLens aLens.to aHash
    lpEventFilter %= HM.insertWith (++) hashVal [(filter', act)]

onButton :: PadButton -> Filter user -> LambdaPad user ()
         -> GameWriter user ()
onButton = onHash buttonHash

onButtonPress :: PadButton -> Filter user -> LambdaPad user ()
              -> GameWriter user ()
onButtonPress but filter' = onButton but $
    with but Pressed && filter'

onButtonRelease :: PadButton -> Filter user -> LambdaPad user ()
                -> GameWriter user ()
onButtonRelease but filter' = onButton but $ 
    with but Released && filter'

onDPad :: Filter user -> LambdaPad user () -> GameWriter user ()
onDPad = onHash dpadHash dpad

onDPadDir :: Direction -> Filter user -> LambdaPad user ()
       -> GameWriter user ()
onDPadDir dir' filter' = onDPad $ with dpad dir' && filter'

onTrigger :: PadTrigger -> Filter user -> LambdaPad user ()
          -> GameWriter user ()
onTrigger = onHash triggerHash

onStick :: PadStick -> Filter user -> LambdaPad user ()
        -> GameWriter user ()
onStick = onHash stickHash

onTick :: LambdaPad user () -> GameWriter user ()
onTick = GameWriter . (lpOnTick %=) . flip (>>)

-- Running

withLambdaPadInner :: LambdaPadInner user a -> LambdaPadMvar user a
withLambdaPadInner act = do
    lambdaPadData <- liftIO . takeMVar =<< get
    (val, lambdaPadData') <- liftIO $ runStateT act lambdaPadData
    liftIO . flip putMVar lambdaPadData' =<< get
    return val

data LambdaPadConfig = LambdaPadConfig
    { padConfigs :: [PadConfig]
    , GameConfigs :: [GameConfig]
    , speedConfig :: Float
    }

-- | Runs LambdaPad with the given configuration.  This starts it in a
-- background thread, but provides the user with a 'Stop' that can be used to
-- stop. 
--
-- Note that this calls SDL.initialize and SDL.quit.  This library is not
-- intended to be used within other SDL applications.

successLambdapad :: LambdaPadConfig -> IO Stop
successLambdapad (LambdaPadConfig{..}) = do

startLambdapad :: Float -> PadConfig -> GameConfig user -> IO Stop
startLambdapad speed padConfig
                   (GameConfig{newUserData, onStop, onEvents}) = do
    SDL.initialize [SDL.InitJoystick]
    numSticks <- SDL.numJoysticks
    joysticks <- SDL.availableJoysticks
    aStop <- if numSticks > 0
      then do
        userData <- newUserData
        joystick <- SDL.openJoystick $ V.head joysticks
        lambdaPadData <- execStateT (runGameWriter onEvents) $ LambdaPadData
            { _lpUserData = userData
            , _lpJoystick = joystick
            , _lpPadConfig = padConfig
            , _lpEventFilter = HM.empty
            , _lpPad = neutralPad
            , _lpOnTick = return ()
            , _lpSpeed = speed
            }
        mvarLambdaPadData <- newMVar $ lambdaPadData
        eventLoop <- initEventLoop mvarLambdaPadData
        tickLoop <- initTickLoop mvarLambdaPadData
        return $ mconcat [tickLoop, eventLoop, cleanUpUser mvarLambdaPadData]
      else return mempty
    return $ mappend aStop $ Stop SDL.quit
  where cleanUpUser mvarLambdaPadData = Stop $ do
            lambdaPadData <- takeMVar mvarLambdaPadData
            onStop $ lambdaPadData^.lpUserData

type LoopIn m = m () -> m ()

runLoopIn :: MonadIO io => (LoopIn io -> IO ()) -> IO (Stop, ThreadId)
runLoopIn acquire = do
    mvarStop <- newMVar ()
    tid <- forkIO $ acquire $ loopIn mvarStop
    return (Stop $ stopLoop mvarStop, tid)
  where
    loopIn :: MonadIO io => MVar () -> LoopIn io
    loopIn mvarStop act = do
        act
        liftIO $ yield
        aStop <- liftIO $ isEmptyMVar mvarStop
        if aStop
          then liftIO $ putMVar mvarStop ()
          else loopIn mvarStop act

stopLoop :: MVar () -> IO ()
stopLoop mvarStop = do
    takeMVar mvarStop -- Signal readLoop
    takeMVar mvarStop -- Wait for readLoop

initEventLoop :: MVar (LambdaPadData user) -> IO Stop
initEventLoop mvarLambdaPadData = do
    (eventLoop, _) <- runLoopIn $ \loopIn ->
        flip evalStateT mvarLambdaPadData $ loopIn listenEvent
    return eventLoop

listenEvent :: LambdaPadMvar user ()
listenEvent = SDL.waitEventTimeout 1000 >>=
    maybe (return ()) (withLambdaPadInner . listen')
  where
    listen' :: SDL.Event -> LambdaPadInner user ()
    listen' SDL.Event{SDL.eventPayload} = do
        mbHash <- case eventPayload of
          SDL.JoyButtonEvent (SDL.JoyButtonEventData
            {SDL.joyButtonEventButton, SDL.joyButtonEventState}) -> do
              on <- buttonConfig <$> use lpPadConfig
              mbBut <- case joyButtonEventState of
                0 -> runLambdaPadState $ on joyButtonEventButton Released
                1 -> runLambdaPadState $ on joyButtonEventButton Pressed
                _ -> do
                  liftIO $ putStrLn $ -- TODO: actual logging
                      "Unrecognized button state: " ++
                      show joyButtonEventState
                  return Nothing
              return $ buttonHash <$> mbBut
          SDL.JoyHatEvent (SDL.JoyHatEventData
            {SDL.joyHatEventHat, SDL.joyHatEventValue}) -> do
              on <- dpadConfig <$> use lpPadConfig
              (fmap.fmap) dpadHash $ runLambdaPadState $
                  on joyHatEventHat joyHatEventValue
          SDL.JoyAxisEvent (SDL.JoyAxisEventData
            {SDL.joyAxisEventAxis, SDL.joyAxisEventValue}) -> do
              on <- axisConfig <$> use lpPadConfig
              mbEiStickTrig <- runLambdaPadState $
                  on joyAxisEventAxis joyAxisEventValue
              return $ flip fmap mbEiStickTrig $ \eiStickTrig -> do
                  case eiStickTrig of
                    Left stick -> stickHash stick
                    Right trig -> triggerHash trig
          _ -> return Nothing
        eventFilter <- use lpEventFilter
        whenJust (mbHash >>= flip HM.lookup eventFilter) evaluateFilter

evaluateFilter :: [(Filter user, LambdaPad user ())] -> LambdaPadInner user ()
evaluateFilter [] = return ()
evaluateFilter ((filter', act):rest) = do
    doAct <- fmap (runFilter filter') get
    if doAct
      then runLambdaPad act
      else evaluateFilter rest

initTickLoop :: MVar (LambdaPadData user) -> IO Stop
initTickLoop mvarLambdaPadData = do
    mvarStop <- newMVar ()
    _ <- SDL.addTimer 0 $ listenTick mvarStop mvarLambdaPadData
    return $ Stop $ stopLoop mvarStop

listenTick :: MVar () -> MVar (LambdaPadData user) -> SDL.TimerCallback
listenTick mvarStop mvarLambdaPadData _ = do
    startTime <- SDL.ticks
    lambdaPadData <- takeMVar mvarLambdaPadData
    lambdaPadData' <- execStateT (lambdaPadData^.lpOnTick.to runLambdaPad)
        lambdaPadData
    putMVar mvarLambdaPadData lambdaPadData'

    let interval = lambdaPadData'^.lpSpeed
    aStop <- isEmptyMVar mvarStop
    if aStop
      then do
        liftIO $ putMVar mvarStop ()
        return SDL.Cancel
      else do
        endTime <- liftIO $ SDL.ticks
        return $ SDL.Reschedule $
            max 1 $ floor (1000 / interval) - (endTime - startTime)