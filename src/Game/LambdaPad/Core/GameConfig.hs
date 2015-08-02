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
  , onEvent
  , onButton
  , onButtonPress
  , onButtonRelease
  , onDPad
  , onDPadDir
  , onTrigger
  , onStick
  , onTick
  , module Game.LambdaPad.Core
  ) where

import Game.LambdaPad.Core
import Game.LambdaPad.Core.Internal
