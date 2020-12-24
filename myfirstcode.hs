import Data.List
import Data.Char
import Text.Read

main = do
    putStrLn "The base?"
    base <- getLine
    putStrLn "The height?"
    height <- getLine
    let area = read base * read height / 2
    putStrLn ("The area of that triangle is " ++ (show area))

factorial 0 = 1
factorial n = n * factorial (n-1)

power x 0 = 1
power x y = power x (y-1) * x

log2 x = go x 0
    where
        go x t
            | x >= 2 = go (x/2) (t+1)
            | otherwise = t

myRepl :: Int -> a -> [a]
myRepl 0 x = []
myRepl count x = x : myRepl (count-1) x

myGet :: [a] -> Int -> a
myGet (x:xs) 0 = x
myGet (x:xs) i = myGet xs (i-1)

myZip :: [a] -> [b] -> [(a, b)]
myZip [] x = []
myZip x [] = []
myZip (a:as) (b:bs) = (a, b) : myZip as bs

myLength :: [a] -> Int
myLength xs = go xs 0
    where
        go :: [a] -> Int -> Int
        go [] t = t
        go (x:xs) t = go xs (t+1)

takeInt :: [Int] -> Int -> [Int]
takeInt _ 0 = []
takeInt [] _ = error "no more items"
takeInt (x:xs) i = x : takeInt xs (i-1)

dropInt :: [Int] -> Int -> [Int]
dropInt xs 0 = xs
dropInt [] _ = error "no more items"
dropInt (_:xs) i = dropInt xs (i-1)

sumInt :: [Int] -> Int
sumInt [] = 0
sumInt (x:xs) = x + sumInt xs

scanSum :: [Int] -> [Int]
scanSum (x:y:ys) = x : scanSum (x+y : ys)
scanSum xs = xs

diff :: [Int] -> [Int]
diff (x:y:ys) = y-x : diff (y : ys)
diff _ = []

myReverse :: [Int] -> [Int]
myReverse [] = []
myReverse (x:xs) = myReverse xs ++ x:[]

mapReverse :: [Int] -> [Int]
mapReverse = map (0 -)

divisorsOfList :: [Int] -> [[Int]]
divisorsOfList = map divisors

divisors p = [ f | f <- [1..p], p `mod` f == 0 ]

divisorsOfListNegate = map (mapReverse . divisors)

decodeRLE :: [Char] -> [(Int, Char)]
decodeRLE xs = map f (group xs)
    where f x = (length x, head x)

encodeRLE :: [(Int, Char)] -> [Char]
encodeRLE xs = concat (map f xs)
    where f x = replicate (fst x) (snd x)

stringRLE :: [(Int, Char)] -> [Char]
stringRLE xs = concat (map f xs)
    where f x = chr (fst x + ord '0') : snd x : []

myAnd :: [Bool] -> Bool
myAnd (x:xs) = x && myAnd xs
myAnd _ = True

myAndFold :: [Bool] -> Bool
myAndFold = foldr (\x b -> x && b) True

myOr :: [Bool] -> Bool
myOr (x:xs) = x || myOr(xs)
myOr _ = False

myOrFold :: [Bool] -> Bool
myOrFold = foldr (\x b -> x || b) False

myMaximum :: Ord a => [a] -> a
myMaximum = foldr1 max

myReverseFold :: [a] -> [a]
myReverseFold = foldr (\x xs -> xs ++ x:[]) []

myReverseFoldl :: [a] -> [a]
myReverseFoldl = foldl (\xs x -> x:xs) []

myScanr :: (a -> b -> b) -> b -> [a] -> [b]
myScanr f acc [] = [acc]
myScanr f acc (x:xs) = f x (head l) : l
    where l = myScanr f acc xs

myScanrFold :: (a -> b -> b) -> b -> [a] -> [b]
myScanrFold f acc xs = foldr g [acc] xs
    where g x l = f x (head l) : l

myScanl :: (b -> a -> b) -> b -> [a] -> [b]
myScanl f acc [] = [acc]
myScanl f acc (x:xs) = acc : myScanl f (f acc x) xs

myScanlFold :: (b -> a -> b) -> b -> [a] -> [b]
myScanlFold f acc xs = (reverse . foldl g [acc]) xs
    where g l x = f (head l) x : l

factList :: Integer -> [Integer]
-- factList x = scanl1 (\acc x -> acc*x) [1..x]
factList x = scanl1 (*) [1..x]

retainEven :: [Int] -> [Int]
-- retainEven = filter (\n -> mod n 2 == 0)
retainEven ns = [ n | n <- ns , mod n 2 == 0 ]

returnDivisible :: Int -> [Int] -> [Int]
returnDivisible x = filter (\n -> mod n x == 0)

headisbiggerthan5 :: [[Int]] -> [[Int]]
headisbiggerthan5 nss = [ tail ns | ns <- nss , (not . null) ns , head ns > 5 ]

myMap :: (a -> b) -> [a] -> [b]
myMap f xs = [ f x | x <- xs ]

myFilter :: (a -> Bool) -> [a] -> [a]
myFilter f xs = [ x | x <- xs, f x ]

doffes :: [(Int, Int)] -> [Int]
doffes = (map (\p -> fst p * 2)) . filter (\p -> mod (snd p) 2 == 0)

data Anniversary = Birthday String Date -- name year month day
                | Wedding String String Date -- name1 name2 year month day

data Date = Date Int Int Int

showDate :: Date -> String
showDate (Date year month day) = show year ++ "-" ++ show month ++ "-" ++ show day

showAnniversary (Birthday name date) =
    name ++ " born " ++ showDate date
showAnniversary (Wedding name1 name2 date) =
    name1 ++ " bmarried " ++ name2 ++ " on " ++ showDate date

fakeIf condition t f =
    case condition of
        True -> t
        False -> f

for :: a -> (a -> Bool) -> (a -> a) -> (a -> IO ()) -> IO ()
for i p f job = do
    if p i
        then do
            job i
            for (f i) p f job
        else return ()

sequenceIO :: [IO a] -> IO [a]
sequenceIO [] = return []
sequenceIO (io:ios) = do
    x <- io
    xs <- sequenceIO ios
    return (x:xs)

mapIO :: (a -> IO b) -> [a] -> IO [b]
mapIO _ [] = return []
mapIO f (x:xs) = do
    y <- f x
    ys <- mapIO f xs
    return (y:ys) 

myCurry :: ((a, b) -> c) -> a -> b -> c
myCurry f x y = f (x, y)

myUnCurry :: (a -> b -> c) -> (a, b) -> c
myUnCurry f (x, y) = f x y

myConst :: a -> b -> a
myConst x _ = x

-- uncurry const = fst
-- curry fst = const
-- curry swap = f x y where f a b = (b, a)

foldlByFoldr :: (b -> a -> b) -> b -> [a] -> b
foldlByFoldr f zero = foldr (flip f) zero . reverse

-- coolFoldl with pointfree
coolFoldl :: (b -> a -> b) -> b -> [a] -> b
coolFoldl f acc = ($ acc) . foldr (flip (.)) id . map (flip f)
-- not pointfree
-- coolFoldl f acc xs = foldr (flip (.)) id (map (flip f) xs) acc

doGuessing num = do {
    putStrLn "Enter your guess:";
    guess <- getLine;
    case compare (read guess) num of {
        LT -> do {
            putStrLn "Too low!";
            doGuessing num;
        };
        GT -> do {
            putStrLn "Too high!";
            doGuessing num;
        };
        EQ -> putStrLn "You Win!";
    };
};

getNumber = do
    x <- getLine
    case readMaybe x :: Maybe Double of
        Just num -> putStrLn $ show num
        Nothing -> putStrLn "Not a double!"

getSum = do
    putStrLn "Write the first one:"
    x <- getLine
    putStrLn "Second one:"
    y <- getLine
    case (readMaybe x :: Maybe Double, readMaybe y :: Maybe Double) of
        (Just n, Just m) -> putStrLn $ "sum of them is... " ++ show (n+m)
        otherwise -> do
            putStrLn "Not a number! Retrying..."
            getSum

interactiveConcatenating :: IO ()
interactiveConcatenating = do
    putStrLn "Choose two strings:"
    s <- (++) <$> getLine <*> (take 3 <$> getLine)
    putStrLn "Let's concatenate them:" *> putStrLn s

-- Board 정의
type Board = Int

-- 다음 턴의 Board 선택지 얻는 함수
-- +1과 *2 두 가지 선택지, 10이 넘어가면 무효
nextConfigs :: Board -> [Board]
nextConfigs x
    | x < 6 = [x+1, x*2]
    | x < 10 = [x+1]
    | otherwise = []

-- 모든 가능한 경우의 수를 재귀적으로 구하자
getAllConfigs :: Board -> [Board]
getAllConfigs x = f [x] where
    f [] = []
    f xs = xs >>= (\y -> y : (f $ nextConfigs y))