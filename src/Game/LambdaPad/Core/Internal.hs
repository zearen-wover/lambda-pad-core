{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}
module Game.LambdaPad.Core.Internal where

import Control.Applicative ( Applicative, (<$>), (<|>) )
import Control.Concurrent
  ( ThreadId, forkIO, yield )
import Control.Concurrent.MVar
  ( MVar, newMVar, isEmptyMVar, takeMVar, putMVar )
import Control.Monad.Reader ( ReaderT, runReaderT )
import Control.Monad.Reader.Class ( MonadReader, ask, local )
import Control.Monad.State.Strict
    ( StateT, evalStateT, execStateT, runStateT )
import Control.Monad.State.Class ( MonadState, get, put )
import Control.Monad.Trans ( MonadIO, liftIO )
import Data.Int ( Int16 )
import Data.Monoid ( Monoid, (<>), mempty, mappend, mconcat )
import Data.Word ( Word8 )

import Data.Algebra.Boolean ( Boolean(..) )
import Control.Lens
    ( Getting, ALens', (.=), (%=), (%%=), (^.), cloneLens, to, use, view )
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
    { runPadState :: StateT Pad (ReaderT SDL.Joystick IO) a }
  deriving (Monad, MonadIO, Functor, Applicative)

newtype GameWriter user a = GameWriter
    { runGameWriter :: LambdaPadInner user a }
  deriving (Monad, Functor, Applicative)

data PadConfig = PadConfig
    { padShortName :: String
    , padName :: String
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

-- | Since 'getSpeed' is typically large, doing an integral action every tick
-- will result in alternating between doing nothing and doing that action e.g.
-- 60 times per second.  This is rarely desired, so this allows one to smooth
-- the action by tracking the residual, or fractional part in user state.
withResidual
    :: Float -- ^ The dead zone.
    -> Float -- ^ The speed in units per second.
    -> ALens' user Float -- ^ The lens to the residual.
    -> Getting Float Pad Float -- ^ The getter to the axis displacement.
    -> LambdaPad user Int
withResidual deadZone unitSpeed residual axis = do
    displacement <- view $ axis.to sqSign
    tickSpeed <- LambdaPad $ use lpSpeed
    if abs displacement < deadZone
      then return 0
      else cloneLens residual %%=
          (splitIntFrac.(+displacement * unitSpeed / tickSpeed))
  where splitIntFrac val = (intVal, val - fromIntegral intVal)
          where intVal = truncate val :: Int
        sqSign x' = signum x'*x'*x'

newtype DPadButton user = DPadButton
    (Direction, Filter user, LambdaPad user (), LambdaPad user ())

dPadButton
  :: Direction -- ^ The direction "button".
  -> Filter user -- ^ A filter for the event.
  -> LambdaPad user () -- ^ What to do on press.
  -> LambdaPad user () -- ^ What to do on release.
  -> DPadButton user
dPadButton dir' filter' onDo unDo = DPadButton (dir', filter', onDo, unDo)

-- | This allows one to pretend that the DPad directions are like buttons.
-- It is unwise to have a 'DPadButton' event on 'C'.
withDPad
  :: ALens' user (LambdaPad user ())
  -- ^ A lens into user state for what to do when the direction is "released".
  -> [DPadButton user] -- ^ The event handlers.
  -> GameWriter user ()
withDPad undoLens dPadButtons = do
    onDPadDir C true $ use (cloneLens undoLens) >>= id >>
        (cloneLens undoLens .= return ())
    mapM_ addDPadButton dPadButtons
  where addDPadButton (DPadButton (dir', filter', onDo, unDo)) = do
            onDPadDir dir' filter' $ do
              cloneLens undoLens .= unDo
              onDo

withStick
  :: ALens' user Bool -- ^ A lens to whether the stick was displaced.
  -> PadStick -- ^ The stick to watch.
  -> StickFilter -- ^ The condition of the stick to watch for.
  -> Filter user
  -- ^ An additional, optional filter.  This only applies when determing whether
  -- the stick was displaced.
  -> (LambdaPad user (), LambdaPad user ()) 
  -- ^ The action to perform when displaced and when returned to neutral resp.
  -> GameWriter user ()
withStick displacedLens stick stickFilter' filter' (onDo, unDo) = do
    onStick stick (stickFilter && not isDisplaced && filter') $ do
      onDo
      cloneLens displacedLens .= True
    onStick stick (not stickFilter && isDisplaced) $ do
      unDo
      cloneLens displacedLens .= False
  where isDisplaced = whenUser (^.cloneLens displacedLens)
        stickFilter = with stick stickFilter'

withTrigger
  :: Float -- ^ The dead zone.
  -> ALens' user Bool -- ^ A lens to whether the trigger was displaced.
  -> PadTrigger -- ^ The trigger to watch.
  -> Filter user
  -- ^ An additional, optional filter.  This only applies when determing whether
  -- the stick was displaced.
  -> (LambdaPad user (), LambdaPad user ()) 
  -- ^ The action to perform when displaced and when returned to neutral resp.
  -> GameWriter user ()
withTrigger deadZone displacedLens trigger filter' (onDo, unDo) = do
    onTrigger trigger (triggerFilter && not isDisplaced && filter') $ do
      onDo
      cloneLens displacedLens .= True
    onTrigger trigger (not triggerFilter && isDisplaced) $ do
      unDo
      cloneLens displacedLens .= False
  where isDisplaced = whenUser (^.cloneLens displacedLens)
        triggerFilter = with trigger $ Pull (>deadZone)

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

instance MonadReader SDL.Joystick PadState where
  ask = PadState $ ask
  local f = PadState . local f . runPadState

runLambdaPadState :: PadState a -> LambdaPadInner user a
runLambdaPadState padState = do
    oldPad <- use lpPad
    joystick <- use lpJoystick
    (val, newPad) <- liftIO $
        flip runReaderT joystick $ flip runStateT oldPad $ runPadState padState
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

newtype PadConfigSelector = PadConfigSelector
    { runPadConfigSelector :: SDL.JoystickDevice -> Maybe PadConfig }

instance Monoid PadConfigSelector where
  mempty = PadConfigSelector $ const Nothing
  mappend (PadConfigSelector s1) (PadConfigSelector s2) =
      PadConfigSelector $ \t -> s1 t <|> s2 t

padConfigByDefault :: PadConfig -> PadConfigSelector
padConfigByDefault = PadConfigSelector . const . Just

startLambdaPad 
  :: PadConfigSelector -> Int -> GameConfig user -> Float -> IO Stop
startLambdaPad padConfigSelector joyIndex gameConfig speed = do
    SDL.initialize [SDL.InitJoystick]
    joysticks <- SDL.availableJoysticks
    if V.length joysticks > joyIndex
      then let joyDevice = joysticks V.! joyIndex 
           in case runPadConfigSelector padConfigSelector $ joyDevice of
                Just padConfig -> fmap (<> Stop SDL.quit) $
                    rawLambdaPad joyDevice padConfig gameConfig speed
                Nothing -> do
                    SDL.quit
                    fail $ "lambda-pad: Failed to load pad config for " ++
                        show (SDL.joystickDeviceName joyDevice)
      else do
          SDL.quit
          fail $ "lambda-pad: Failed to open joystick at index " ++
              show joyIndex

rawLambdaPad 
  :: SDL.JoystickDevice -> PadConfig -> GameConfig user -> Float
  -> IO Stop
rawLambdaPad joystickDevice padConfig
             (GameConfig{newUserData, onStop, onEvents}) speed = do
    userData <- newUserData
    joystick <- SDL.openJoystick joystickDevice
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
