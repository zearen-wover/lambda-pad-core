# lambdapad

This allows a user to write an application based on events for a game pad.  Note
that this is not a game library.  The original use case for this was as yet
another game pad to keyboard and mouse event utility, but I left the API very
general.

Note that this is not for muggles.  All configuration will be done via Dyre.
Knowledge of Haskell is assumed.  The profiles are launched via the
command line.

# Documentation

To ease configuration, there's a layer of indirection between the pad
configuration and game configuration.  The actual buttons and axes on the pad
are translated into a set of buttons on the typical controller which are given
symbols.  The game configurations are then based on events and conditions of
these symbols.

## Pad profiles

TODO: Fill this out.

This maps the pad hardware to the logical buttons.

TODO: Write a utility to help write these files.

## Game profiles

TODO: Fill this out.

This is a state monad that tracks logical button presses and then sends commands
when certain conditions are met (e.g. A and B are pressed which would be
rendered as `with a Pressed && with b Pressed`).  There is also a periodic
background thread that can have actions added to it.

# Building

If for some strange reason you want to build this:

    git clone http://github.com/zearen-wover/lambda-pad.git
    git clone http://github.com/haskell-game/sdl2.git
    cd lambda-pad
    cabal sandbox init
    cabal sandbox add-source ../sdl2
    cabal install --dependencies-only
    cabal configure
    cabal build
    ./dist/build/lambdapad/lambdapad
