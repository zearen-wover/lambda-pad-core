module Game.LambdaPad.GameConfig
  ( GameConfig(..), GameWriter
  , LambdaPad
  , isPad
  , getSpeed
  , withResidual
  , Filter
  , whenPad
  , whenUser
  , WithFilter(with)
  , Pull(..)
  , StickFilter(..)
  , onButton
  , onButtonPress
  , onButtonRelease
  , onDPad
  , onDPadDir
  , onTrigger
  , onStick
  , onTick
  , module Game.LambdaPad
  ) where

import Game.LambdaPad hiding ( lambdaPad, Stop, stop )
import Game.LambdaPad.Internal
