module Main where

import Data.Char (toUpper)
import Control.Monad

main = putStrLn "Write your string: " >> fmap shout getLine >>= putStrLn

shout = map toUpper