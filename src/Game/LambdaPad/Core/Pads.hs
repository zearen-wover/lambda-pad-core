module Game.LambdaPad.Core.Pads
    ( allKnownPads
    , padConfigByName
    , padConfigByShortName
    ) where

import Data.Maybe ( listToMaybe )

import Data.Text ( unpack )
import SDL ( joystickDeviceName )

import Game.LambdaPad.Core.Internal ( PadConfigSelector(..) )
import Game.LambdaPad.Core.PadConfig ( PadConfig(..) )
import Game.LambdaPad.Pads.F310 ( f310 )


allKnownPads :: [ PadConfig ]
allKnownPads = [ f310 ]

padConfigByName :: PadConfigSelector
padConfigByName = PadConfigSelector $ \joystick -> listToMaybe $
    filter (padHasJoystickName joystick) allKnownPads
  where padHasJoystickName joystick pad =
            (padName pad ==) $ unpack $ joystickDeviceName joystick
    
padConfigByShortName :: String -> PadConfigSelector
padConfigByShortName name = PadConfigSelector $ const $
    listToMaybe $ filter ((name==) . padShortName) allKnownPads
