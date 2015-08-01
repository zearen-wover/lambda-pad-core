# lambda-pad-core

This is a library that allows one to launch IO actions based on events from and
the state of a game-pad.  This is the core llibrary in case you just want to use
the event and filter functionality with the executable.  Most of the
documentation is associated with the main executable:
[lambda-pad](https://github.com/zearen-wover/lambda-pad).  Note this that
provides `Game.LambdaPad.GameConfig`, `Game.LambdaPad.PadConfig` and
`Game.LambdaPad.Pads`, but with `Core` prepended to the innermost module name.
E.g. `Game.LambdaPad.Core.Pads`.

# Building and Installing

If for some strange reason you want to build this manually:

    git clone http://github.com/haskell-game/sdl2.git
    cd sdl2
    cabal configure
    cabal build
    cabal install
    cd ..
    git clone http://github.com/zearen-wover/lambda-pad-core.git
    cd lambda-pad
    cabal configure
    cabal build
    cabal install
