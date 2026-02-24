#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>
#include <immintrin.h>
#include <immintrin.h>

extern unsigned char* convolution2DInASM1(int width, int height, int channels, float* img, unsigned char* outputImg, float* kernel);
// ورژن کندتر
// unsigned char* convolution2DInASM(int width, int height, int channels,
//                                   float* img
//                                   , unsigned char* outputImg, float kernel[]) {

//     float tempKernel[32];
//     float temp_image25Comp[32];
// // کپی کرنل و پر کردن با صفر
//     for(int i = 0 ; i < 25 ;i++){
//         tempKernel[i] = kernel[i];
//     }

//     for(int i = 25 ; i < 32; i++){
//         tempKernel[i] = 0.0f;
//         temp_image25Comp[i] = 0.0f;
//     }
//                 __m256 v_ker1 = _mm256_loadu_ps(&tempKernel[0]);
//                 __m256 v_ker2 = _mm256_loadu_ps(&tempKernel[8]);
//                 __m256 v_ker3 = _mm256_loadu_ps(&tempKernel[16]);
//                 __m256 v_ker4 = _mm256_loadu_ps(&tempKernel[24]);

//     for(int i = 2 ; i < height-2 ; i++){
//         for(int j = 2 ; j < width-2; j++){
//             for(int c = 0; c < channels; c++){
//                 if(c == 3){
//                     outputImg[(i * width + j) * channels + c] = (char)img[(i * width + j) * channels + c];
//                     continue;
//                 }
//                 int counter = 0 ;
//                 for(int h = i-2; h <= i+2; h++ ){
//                     for(int w = j-2; w <= j+2; w++){
//                              int imgidx = (h*width+w)*channels+c;
//                             temp_image25Comp[counter++] = (float)img[imgidx];
//                     }
//                 }
//                 // بارگذاری ایمن داده‌ها از حافظه به رجیستر (Load Unaligned)

//                 __m256 v_img1 = _mm256_loadu_ps(&temp_image25Comp[0]);
//                 __m256 v_img2 = _mm256_loadu_ps(&temp_image25Comp[8]);
//                 __m256 v_img3 = _mm256_loadu_ps(&temp_image25Comp[16]);
//                 __m256 v_img4 = _mm256_loadu_ps(&temp_image25Comp[24]);

//                 // ضرب
//                 __m256 m1 = _mm256_mul_ps(v_img1, v_ker1);
//                 __m256 m2 = _mm256_mul_ps(v_img2, v_ker2);
//                 __m256 m3 = _mm256_mul_ps(v_img3, v_ker3);
//                 __m256 m4 = _mm256_mul_ps(v_img4, v_ker4);
//                 // جمع
//                 __m256 sum1 = _mm256_add_ps(m1, m2);
//                 __m256 sum2 = _mm256_add_ps(m3, m4);
//                 __m256 total = _mm256_add_ps(sum1, sum2);
//                 float temp[8];
//                 _mm256_storeu_ps(temp, total);
//                 float result = 0.0f;
//                 for(int m = 0 ; m < 8 ; m++){
//                     result += temp[m];
//                 }
//                 outputImg[(i * width + j) * channels + c] = (unsigned char)result;
//             }
//         }
//     }
//     return outputImg;

// }
// ورژن سریعتر
unsigned char* convolution2DInASM(
    int width, int height, int channels,
    float* img,unsigned char* outputImg,
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
                        outputImg[(i*width + j+k)*channels + c] = (unsigned char)img[(i*width + j+k)*channels + c];
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
                    outputImg[(i*width + j+k)*channels + c] = (unsigned char)results[k]; // ذخیره ی نتیجه در خروجی و کست کردن ان با کارکتر از اعشاری
            }
        }
    }

    return outputImg;
}


// convolution2D in c
unsigned char* convolution2DInC(int width,int height
    ,int channels,unsigned char *img
    ,unsigned char* outputImg,float kernel[]){ 
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
                    outputImg[(i * width + j) * channels + c]=(unsigned char)sum;//قرار دادن در ایندکس متناظر خروجی
                }
            }
        }
    return outputImg;
}
//تابع اصلی
int main(){
    int width;
    int height;
    int channels;//تعداد کانال‌های رنگی (مثلاً 3 برای RGB و 4 برای RGBA)
    char *imagePath = "Image.jpg";
    // مشخصات عکس در ۴ متغیر بالا ذخیره میشود
    //طول . عرض و وکانال و مسیر فایل عکس

    unsigned char *img = stbi_load(imagePath,&width,&height,&channels,0); //لود تصویر به صورت ارایه یک بعدی بصورت [rgbrgbrgbrgb...]

    if(img == NULL){ // عکسی وجود نداره
        printf("this image does not exist.");
        return 1;
    }
    float *imgFloat = (float*)malloc(width * height * channels * sizeof(float));
    for(int i = 0; i < width * height * channels; i++) {
        imgFloat[i] = (float)img[i];
    }

    unsigned char *outputImage = malloc(width*height*channels); // رزرو فضا برای خروجی

    float kernel[25] = { // کرنل ۵ در ۵ برای گرفتن میانگین از پیکسل های اطراف پیکسل اصلی برای blur
        0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
        0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
        0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
        0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
        0.04f, 0.04f, 0.04f, 0.04f, 0.04f
    };
    struct timespec start, end;
    double total_c = 0.0;
    double total_asm = 0.0;
    int runs = 10;

    // تست نسخه C
    for(int r = 0; r < runs; r++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        convolution2DInC(width, height, channels, img, outputImage, kernel);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_c += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    // تست نسخه ASM
    for(int r = 0; r < runs; r++){
        clock_gettime(CLOCK_MONOTONIC, &start);
        convolution2DInASM(width, height, channels, imgFloat, outputImage, kernel);
        // convolution2DInASM1(width, height, channels, imgFloat, outputImage, kernel);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_asm += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    printf("Average Time C:   %f seconds\n", total_c / runs);
    printf("Average Time ASM: %f seconds\n", total_asm / runs);


    stbi_write_png("outoutImage.png",width,height,channels,outputImage,width*channels);//نوشتن خروجی در یک تصویر
    
    printf("We successfully could have blurred the image.\n"); //نتیجه

    stbi_image_free(img); //ازاد سازی حافظه ی  ارایه ی load عکس
    free(outputImage);//ازاد سازی حافظه رزرو شده ی عکس

    return 0;

}