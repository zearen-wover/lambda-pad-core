module Game.LambdaPad.Core.Pads
  ( allKnownPads
  , f310
  ) where

import Game.LambdaPad.Core.PadConfig ( PadConfig )
import Game.LambdaPad.Pads.F310 ( f310 )
import Game.LambdaPad.Pads.XBox ( xbox )

allKnownPads :: [ PadConfig ]
allKnownPads = [ f310, xbox ]
