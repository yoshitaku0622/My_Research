#include <stdio.h>
#define LOWER  0
#define UPPER  300
#define STEP  10
#define PI 3.14

int main(void)
{
    //fahrからcelsius....................
    /*
    float fahr;
    float celsius;

    printf("fahr\t\tcelsius\n");

    fahr = LOWER;
    while(fahr  <= UPPER)
    {
        celsius = 5.0 / 9.0 * ( fahr - 32.0 );
        printf("%f\t%f\n", fahr, celsius);
        fahr += STEP;
    }
    */



    //celsiusからfahr.....................
    /*
    printf("celsius\tfahr");

    celsius = UPPER;
    while(celsius >= LOWER)
    {
       fahr = celsius * 9.0 / 5.0 +32.0;
        printf("%d\t%d\n", celsius, fahr);
        celsius -= STEP;
    }
   */



    //和・差・積・商の計算......................
    /*
    int a = 10;
    int b = 2;
    int wa, sa, seki, shou;

    wa = a + b;
    sa = a - b;
    seki = a * b;
    shou = a / b;

    printf("和：%d\t差:%d\t積:%d\t商:%d\n", wa, sa, seki, shou);
    */



   //円面積、円周................................
    int r = 5;
    printf("面積:%f\t円周:%f\n", 2 * r * PI, r * r * PI);
}