module Game.LambdaPad.Core
  ( ButtonState(Pressed, Released)
  , Button
  -- | A lens into whether the button is pressed.
  , buttonState
  , Direction (C, N, NE, E, SE, S, SW, W, NW)
  , DPad
  -- | A lens into the 'Direction' a 'DPad' is pressed.
  , dir
  , Trigger
  -- | A lens into how far a trigger has been pressed, where 0.0 is neutral and
  -- 1.0 is fully depressed.
  , pull
  , Stick
  -- | A lens into the horizontal displacement of a 'Stick', where 0.0 is
  -- neutral, -1.0 is fully W, and 1.0 is fully E.  Think of it as the X axis
  -- bounded on [-1.0, 1.0]
  , horiz
  -- | A lens into the vertical displacement of a 'Stick', where 0.0 is neutral,
  -- -1.0 is fully S, and 1.0 is fully N.  Think of it as the X axis bounded on
  -- [-1.0, 1.0]
  , vert
  , tilt, push
  , Pad
  , PadButton
  , a, b, x, y
  , lb, rb, ls, rs
  , back, start, home
  , PadDPad
  , dpad
  , PadTrigger
  , leftTrigger, rightTrigger
  , PadStick
  , leftStick, rightStick
  ) where

import Game.LambdaPad.Core.Internal
