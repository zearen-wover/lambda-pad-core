module Game.LambdaPad.Core.PadConfig
  ( PadConfig(..), PadState
  , simpleButtonConfig
  , simpleDPadConfig
  , simpleAxisConfig
  , triggerConfig
  , horizStickConfig
  , vertStickConfig
  , module Game.LambdaPad
  ) where

import Game.LambdaPad.Core hiding ( startLambdaPad, Stop, stop )
import Game.LambdaPad.Core.Internal
