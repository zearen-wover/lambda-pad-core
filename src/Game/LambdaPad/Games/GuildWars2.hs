{-# LANGUAGE TemplateHaskell #-}
module Game.LambdaPad.Games.GuildWars2 ( guildWars2 ) where

import Control.Monad ( replicateM_, when )
import Control.Monad.IO.Class( liftIO )
import Control.Monad.State.Class( get )
import Control.Lens ( ALens', (^.), (.=), cloneLens, use, _1, _2 )
import Control.Lens.TH ( makeLenses )
import Data.Algebra.Boolean (Boolean(..))
import Prelude hiding ( (&&), (||),  not )

import Game.LambdaPad.GameConfig
import Test.Robot ( Robot, Pressable(press, release), moveBy, tap )
import Test.Robot.Connection ( runRobotWith )

import qualified Graphics.XHB as X
import qualified Graphics.XHB.Gen.Shape as X
import qualified Test.Robot.Types as K

data StickPress = StickPress
    { _nPress :: Bool
    , _ePress :: Bool
    , _sPress :: Bool
    , _wPress :: Bool
    }
makeLenses ''StickPress

data GuildWars2 = GuildWars2
    { gw2SConn :: !X.Connection
    , gw2ScreenSize :: !(Int, Int)
    , gw2MouseSpeed :: !Float
    , _gw2MouseResidual :: !(Float, Float)
    , _gw2ScrollResidual :: !Float
    , _gw2UndoDir :: LambdaPad GuildWars2 ()
    , _gw2LeftStickPress :: !StickPress
    , _gw2Panning :: !Bool
    }
makeLenses ''GuildWars2

guildWars2 :: Float -> GameConfig GuildWars2
guildWars2 mouseSpeed = GameConfig
    { gameName = "gw2"
    , newUserData = do
          xConn <- maybe (fail "Could not connect to X server") return =<<
              X.connect
          screenSize@(width, height) <- getScreenSize xConn
          return $ GuildWars2
              { gw2SConn = xConn
              , gw2ScreenSize = screenSize
              , gw2MouseSpeed = mouseSpeed * fromIntegral (max width height)
              , _gw2MouseResidual = (0, 0)
              , _gw2ScrollResidual = 0
              , _gw2UndoDir = return ()
              , _gw2LeftStickPress = StickPress False False False False
              , _gw2Panning = False
              }
    , onStop = const $ return ()
    , onEvents = do
          buttonAsKey a K._Space true
          buttonAsKey x K._1 true
          buttonAsKey y K._2 true
          buttonAsKey b K._3 true
          buttonAsKey lb K._4 true
          buttonAsKey rb K._5 true

          buttonAsKey lb K._6 shiftMode
          buttonAsKey x K._7 shiftMode
          buttonAsKey y K._8 shiftMode
          buttonAsKey b K._9 shiftMode
          buttonAsKey rb K._0 shiftMode

          onDPadDir C true $ use gw2UndoDir >>= id >> (gw2UndoDir .= return ())
          dirAsKey N [K._F1] true
          dirAsKey E [K._F2] true
          dirAsKey S [K._F3] true
          dirAsKey W [K._F4] true

          dirAsKey N [K._T] shiftMode
          dirAsKey E [K._Tab] shiftMode
          dirAsKey S [K._Ctrl, K._T] shiftMode
          dirAsKey W [K._Shift, K._Tab] shiftMode

          buttonAsKey rs K._F true
          buttonAsKey rs K.rightButton mouseMode
          buttonAsKey rb K.leftButton mouseMode

          buttonAsKey home K._Escape true
          buttonAsKey start K._I true
          buttonAsKey back K._Grave true
          buttonAsKey start K._H shiftMode
          buttonAsKey back K._M shiftMode
          
          buttonAsKey ls K._V true
          stickAsKey leftStick (gw2LeftStickPress.ePress) (Horiz (>0.2))
              K._E true
          stickAsKey leftStick (gw2LeftStickPress.wPress) (Horiz (<(-0.2)))
              K._Q true
          stickAsKey leftStick (gw2LeftStickPress.nPress) (Vert (>0.2))
              K._W $ not (mouseMode && shiftMode)
          stickAsKey leftStick (gw2LeftStickPress.sPress) (Vert (<(-0.2)))
              K._S $ not (mouseMode && shiftMode)

          stickAsKey rightStick gw2Panning (Push (>0.05))
              K.rightButton $ not mouseMode
          -- If we enter mouse mode, we should stop panning.
          onTrigger leftTrigger (mouseMode && whenUser (^.gw2Panning)) $ do
              runRobot $ release K.rightButton
              gw2Panning .= False

          onTick $ do
            moveMouse <- isPad $ mouseMode
            isPanning <- use $ gw2Panning
            when (moveMouse || isPanning) $ do
              mouseSpeed' <- fmap 
                  (if isPanning then const 400 else gw2MouseSpeed) get
              doScroll <- isPad $ shiftMode
              x' <- withResidual 0.05 mouseSpeed' (gw2MouseResidual._1)
                  (rightStick.horiz)
              y' <- withResidual 0.05 mouseSpeed' (gw2MouseResidual._2)
                  (rightStick.vert)
              scroll <- if moveMouse && doScroll
                then withResidual 0.05 10 gw2ScrollResidual (leftStick.vert)
                else return 0
              runRobot $ do
                  moveBy x' (negate y')
                  if scroll >= 0
                    then replicateM_ scroll $ tap K.scrollUp
                    else replicateM_ (negate scroll) $ tap K.scrollDown
    }

runRobot :: Robot a -> LambdaPad GuildWars2 a
runRobot rbt = fmap gw2SConn get >>= liftIO . flip runRobotWith rbt

buttonAsKey :: Pressable key
            => PadButton -> key -> Filter GuildWars2
            -> GameWriter GuildWars2 ()
buttonAsKey but key filter' = do
    onButtonPress but filter' $ runRobot $ press key
    onButtonRelease but filter' $ runRobot $ release key

dirAsKey :: Pressable key
         => Direction -> [key] -> Filter GuildWars2
         -> GameWriter GuildWars2 ()
dirAsKey dir' keys filter' = do
  onDPadDir dir' filter' $ do
      gw2UndoDir .= (mapM_ (runRobot . release) $ reverse keys)
      mapM_ (runRobot . press) keys

stickAsKey :: Pressable key
           => PadStick -> ALens' GuildWars2 Bool -> StickFilter -> key
           -> Filter GuildWars2 -> GameWriter GuildWars2 ()
stickAsKey stick isTilted' stickFilter key filter' = do
    onStick stick (with stick stickFilter &&
                   (not $ whenUser (^.isTilted)) && filter') $ do
        runRobot $ press key
        cloneLens isTilted' .= True
    onStick stick (not (with stick stickFilter) && whenUser (^.isTilted)) $ do
        runRobot $ release key
        cloneLens isTilted' .= False
  where isTilted = cloneLens isTilted'

shiftMode :: Filter user
shiftMode = with rightTrigger $ Pull (>0.2)

mouseMode :: Filter user
mouseMode = with leftTrigger $ Pull (>0.2)

-- TODO: This should be broken out into its own library.
getScreenSize :: X.Connection -> IO (Int, Int)
getScreenSize xConn = do
    receipt <- X.queryExtents xConn $ X.getRoot xConn
    eiReply <- X.getReply receipt
    case eiReply of
      Left someError -> fail $ show someError
      Right reply -> return
          ( fromIntegral $
                X.bounding_shape_extents_width_QueryExtentsReply reply
          , fromIntegral $
                X.bounding_shape_extents_height_QueryExtentsReply reply
          )
