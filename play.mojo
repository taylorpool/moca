from algorithm import vectorize, vectorize_unroll, parallelize
from time import sleep


fn main():
    var test = 0

    fn run[n: Int](arg: Int) -> None:
        print(n, arg)

    vectorize_unroll[4, 2, run](100)
