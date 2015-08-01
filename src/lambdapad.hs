import Game.LambdaPad ( lambdapad, stop )
import Game.LambdaPad.Pads.F310 ( f310 )
import Game.LambdaPad.Games.GuildWars2 ( guildWars2 )

main :: IO ()
main = do
  lpLoop <- lambdapad f310 $ guildWars2 1.0
  _ <- getLine
  stop lpLoop
