#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "weights.h" //وزن های تولید شده توسط پایتون
//convolution 2D in ASM
float* convolution2DInASM(
    int width, int height, int channels,
    float* img,float* outputImg,
    float kernel[])
{
    // این بردار فاصله‌ی هر پیکسل از پیکسل اول را مشخص می‌کند
    __m256i INDEX = _mm256_setr_epi32(
        0, channels, 2*channels, 3*channels, 
        4*channels, 5*channels, 6*channels, 7*channels
    );
    for(int i = 2; i < height - 2; i++){ // پیشمایش پیکسل ها
        for(int j = 2; j < width - 10; j += 8){
            for(int c = 0; c < channels; c++){// پیمایش در rgb
                if(c == 3){ // اگه آلفا وجود داره و کپیش کن بذار تو همون ایندکس
                    for(int k=0;k<8;k++)
                        outputImg[(i*width + j+k)*channels + c] = img[(i*width + j+k)*channels + c];
                    continue;
                }

                __m256 sum = _mm256_setzero_ps(); //sum = [0,0,0,0,0,0,0,0]
                int counter = 0; // شمارنده برای کرنل
                for(int h = -2; h <= 2; h++) // پیمایش در کرنل و ۲۵ پیکسل اطراف هر پیکسل
                {
                    int base = ((i+h)*width + j-2) * channels + c; 
                    //     محاسبه ی پیکسل متناظر هر درایه کرنل و محاسبه ی پایه ان 
                    //چون تصاویر عمدتا  بصورت ار جی بی هست باید بصورت ۳ تا ۳ تا جلو بریم تا برای هر ۸ پیکسل و پیکسل متناظر کرنل بدست بیاوریم
                    for(int w = 0; w < 5 ; w++)
                    {
                        float kernelDer = kernel[counter++];
                        __m256 valuOfKernel = _mm256_set1_ps(kernelDer);//kernelDer = n,value of kernel : [n,n,n,n,n,n,n,n]  
                        
                        __m256 vPixels = _mm256_i32gather_ps(&img[base + w * channels], INDEX, 4);//لود کردن پیکسل های متناظر در یک رجبستر
                        sum = _mm256_fmadd_ps(vPixels, valuOfKernel, sum); //ضرب ۸ پیکسل متنظر در یک درایه کرنل و جمع  ان با  متغیر سام
                    }
                }
                // محاسبه ی ضرب ماتریس کرنل در یکی از کانال های ار جی بی اطراف هر۸ پیکسل همزمان
                float results[8];
                _mm256_storeu_ps(results, sum);//ذخیره نتیجه در یک ارابه و جدا سازی ان

                for(int k=0;k<8;k++)
                    outputImg[(i*width + j+k)*channels + c] = results[k]; // ذخیره ی نتیجه در خروجی و کست کردن ان با کارکتر از اعشاری
            }
        }
    }

    return outputImg;
}


// convolution2D in c
float* convolution2DInC(int width,int height
    ,int channels,float *img
    ,float* outputImg,float kernel[]){ 
        for (int i = 0 ; i<height; i++){ // پیشمایش پیکسل ها
            for(int j = 0 ; j<width; j++){ 
                for(int c = 0 ; c < channels ; c++){ // پیمایش در rgb

                    if(c == 3){  // اگه آلفا وجود داره و کپیش کن بذار تو همون ایندکس
                        outputImg[(i * width + j) * channels + c] = img[(i * width + j) * channels + c];
                        continue;
                    }
                    int kernelidx = 0; //برای ایندکس کرنل, بعد گذر پیکسلی تبدیل به صفر میشود
                    float sum = 0.0;
                    for(int h = i-2; h <= i+2; h++ ){ //پیمایش در ماتریس پنج در پنج اطراف پیکسل
                        for(int w = j-2; w <= j+2; w++){

                            if(h>=0 && w>=0 && h < height && w < width){  //از مرز خارج نشود
                                int imgidx = (h*width+w)*channels+c;//محاسبه ایندکس در پیشمایش پیکسل های بر اساس ارایه یک بعدی
                                sum += kernel[kernelidx]*img[imgidx]; // ضرب مقدار کرنل در مقدار هر عضو ار جی بی در پیکسل موردنظر
                            }
                            kernelidx++;
                        }
                    }
                    //در هر پیمایش مقدار کانولوشن هر عضو ار جی بی برای ماتریس ۵ در ۵ اطراف پیکسل محاسبه میشود
                    outputImg[(i * width + j) * channels + c] = sum;//قرار دادن در ایندکس متناظر خروجی
                }
            }
        }
    return outputImg;
}

// تابع فعال ساز  
// اینجا از این تابع استفاده کردم تا مقادیر منفی حذف شوند و مدل بتواند رفتار غیرخطی داشته باشد. بدون آن، چند لایه پشت‌سرهم هم معادل یک تبدیل خطی ساده می‌شوند
void relu(float* img, int size) {
    for(int i = 0; i < size; i++) {
        if(img[i] < 0) {
            img[i] = 0.0f;
        }
    }
}
// Wx+b
// اینجا مدل بین ۱۰ کلاس رقابت ایجاد می‌کند و هر کلاسی که بیشترین نمره را بگیرد، خروجی نهایی می‌شود.
int fully_connected_and_predict(float* conv_output) {
    float max_score = -999999.0f;
    int best_class = -1;
    
    for(int i = 0; i < 10; i++) {  // پیمایش در ۱۰ کلاس چون ۱۰ رقمه
        float score = fc_bias[i]; //برای اینکه از مبدا نگذرد تابع
        for(int j = 0; j < 28 * 28; j++) {
            score += fc_weight[i][j] * conv_output[j]; //کدوم ویژگی ها برای ما ارزش بیشتری داره
        }
        
        if(score > max_score) {
            max_score = score; //انتخاب اینکه کدوم کلاس بهتره
            best_class = i; //کدوم کلاس
        }
    }
    return best_class;
}
//benchmark
int main() {
    int width, height, channels;//نگهداری متغیر ها در متغیر
    
    int n = 10;
    int accuracyCounter = 0 ;
    for(int i = 0 ; i < n ; i++){
        char filename[64];
        sprintf(filename,"benchmarkDataSet/%d.jpg",i);
        // لود کردن تصویر
        unsigned char *imgData = stbi_load(filename, &width, &height, &channels, 1);
        
        //ایا تونیستم پیدا کنیم تصویر را
        if (imgData == NULL) {
            printf("Error: Could not load image. Make sure 'test5.png' exists.\n");
            return 1;
        }
        //ایا تصویر ۲۸ پیکسل در ۲۸ پیکسل است؟ 
        if (width != 28 || height != 28) {
            printf("Error: Your image is %dx%d. It MUST be exactly 28x28 pixels!\n", width, height);
            stbi_image_free(imgData);
            return 1;
        }

        channels = 1;  //میدانیم قراره عکس ها سیاه سفید باشند
        float* imgFloat = (float*)malloc(width * height * sizeof(float));  //رزرو حافظه برای ورودی  
        float* convOutputC = (float*)calloc(width * height, sizeof(float)); // رزرو برای نتیجه ی عکس در سی 
        float* convOutputASM = (float*)calloc(width * height, sizeof(float)); //رزرو برای نتیجه ی عکس در اسمبلی
        
        // 2. پیش‌پردازش تصویر

        for (int i = 0; i < width * height; i++) {
            float val = imgData[i] / 255.0f; //تبدیل مقادیر پیکسلی عکس از ۰ تا ۲۵۵ به ۰ تا۱
            imgFloat[i] =  val; //اینکار را برای این انجام میدهیم تا اورفلو رخ ندهد
        }
        

        clock_t start_time, end_time; //متغیر برای محاسبه تایم
        double time_c, time_asm;

        printf("Processing %dx%d image...\n", width, height);

        //تست سی
        start_time = clock();
        convolution2DInC(width, height, channels, imgFloat, convOutputC, conv_kernel); 
        end_time = clock();
        time_c = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
        
        //تست اسمبلی
        start_time = clock();
        convolution2DInASM(width, height, channels, imgFloat, convOutputASM, conv_kernel);
        end_time = clock();
        time_asm = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

        printf("\n--- Execution Time Comparison ---\n");
        printf("Standard C Convolution: %f seconds\n", time_c);
        printf("AVX/ASM Convolution:    %f seconds\n", time_asm);
        if (time_asm > 0) {
            printf("Speedup:                %.2fx faster!\n", time_c / time_asm);
        }
        printf("---------------------------------\n");

        // neural network
        //تابع فعال ساز
        relu(convOutputC, width * height);
        //تابع پیش بینی 
        int predicted_digit = fully_connected_and_predict(convOutputC);
        
        if(predicted_digit == i){
            accuracyCounter++;
        }
        printf("\n>>> The predicted digit is: %d <<<\n", predicted_digit);
        

        //پاک کردن رزروی ها
        stbi_image_free(imgData);
        free(imgFloat);
        free(convOutputC);
        free(convOutputASM);
    }
    printf("this model's accuracy is %0.2f%%\n",(float)accuracyCounter/10);
    return 0;
}

