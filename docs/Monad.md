# Monad

모나드 기초 이해 복습

https://wikidocs.net/1471



**Monad Laws**

```haskell
m >>= return     =  m                        -- 우단위원의 법칙(right unit)
return x >>= f   =  f x                      -- 좌단위원의 법칙(left unit)

(m >>= f) >>= g  =  m >>= (\x -> f x >>= g)  -- 결합법칙(associativity)
```

`return`이 일종의 항등원이다.

> 결합법칙은 세미콜론이 그러듯이 bind 연산자 (>>=)가 계산의 순서만 신경쓸 뿐 그 중첩 구조는 고려하지 않음을 보장한다.
>
> The law of associativity makes sure that (like the semicolon) the bind operator `(>>=)` only cares about the order of computations, not about their nesting;

?? 잘 이해 안됨



**>>= (bind operator)**

`>>= :: Monad m => m a -> (a -> m b) -> m b`

bind 연산자는 할당과 함께 do 블럭 안의 ;와 교환할 수 있다.

`x <- foo; return (x+3)` == `foo >= (\x -> return (x+3))`

즉 Monad a 표현식은 명령형 언어에서 a를 반환하는 명령으로 생각할 수 있다.



**Monad에 대한 관점**

Monad a에 대해 생각할 수 있는 관점:

1. `return`과 `join`을 가지는 펑터
2. 펑터이므로, a 타입을 보관하고 있는 컨테이너
3. 명령형 언어에서 a 타입을 반환하는 하나의 명령
4. 명령형 언어로서의 의미 정의

Monad는 Applicative Functor의 서브클래스이다. (하스켈에 정립된지 얼마 안됨)

그러므로 모나드를 정의할 때 `Functor`, `Applicative`, `Monad`순으로 인스턴스화가 바람직하다.

역으로 `return`과 `bind` 정의로 `Monad`를 인스턴스화하고 `liftM`, `ap`, `return`으로 슈퍼클래스를 채울 수 있다.

모나드에 한정하여 `liftM` == `fmap`, `ap` == `<*>`이다.

단순히 레거시 함수라 생각했는데 반대 순서로 인스턴스화하는 등 쓸모가 많아보인다.