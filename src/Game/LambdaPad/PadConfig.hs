module Game.LambdaPad.PadConfig
  ( PadConfig(..), PadState
  , simpleButtonConfig
  , simpleDPadConfig
  , simpleAxisConfig
  , triggerConfig
  , horizStickConfig
  , vertStickConfig
  , module Game.LambdaPad
  ) where

import Game.LambdaPad hiding ( lambdaPad, Stop, stop )
import Game.LambdaPad.Internal
