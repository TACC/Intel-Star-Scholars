#syclcc Check_Device.cpp -O2 -o check
#icpx -fsycl -O2 -Wdeprecated VectorAddA.cpp -o simulate1
#icpx -fsycl Check_Device2.cpp -o check
#syclcc -O2 VectorAddA.cpp -o simulate1
#syclcc -O2 VectorAddB.cpp -o simulate2
#syclcc -O2 VectorAddC.cpp -o simulate3
#icpx -fsycl -O2 -Wdeprecated VectorMultA.cpp -o simulate4
#icpx -fsycl -O2 -Wdeprecated VectorMultB.cpp -o simulate5
#icpx -fsycl -O2 -Wdeprecated VectorMultC.cpp -o simulate6
#icpx -fsycl -O2 -Wdeprecated VectorStencilB.cpp -o simulate8
icpx -fsycl -O2 -Wdeprecated VectorStencilC.cpp -o simulate13
#icpx -fsycl -O2 -Wdeprecated VectorSaxpyA.cpp -o simulate10
#icpx -fsycl -O2 -Wdeprecated VectorSaxpyB.cpp -o simulate11
#icpx -fsycl -O2 -Wdeprecated VectorSaxpyC.cpp -o simulate12
#icpx -fsycl -O2 -Wdeprecated VectorMultATile.cpp -o tile
