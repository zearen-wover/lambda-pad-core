module Game.LambdaPad.Core.GameConfig
  ( GameConfig(..), GameWriter
  , LambdaPad
  , isPad
  , getSpeed
  , withResidual
  , DPadButton, dPadButton
  , withDPad
  , withStick
  , withTrigger
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

import Game.LambdaPad.Core hiding ( lambdaPad, Stop, stop )
import Game.LambdaPad.Core.Internal