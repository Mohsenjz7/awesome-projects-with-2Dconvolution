// 403106013
// square detection


#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>
#include <immintrin.h>
#include <math.h>

#define EDGE_THRESHOLD 120
//تابع پیدا کردن مربع)
int detectSquare(int width, int height, int channels, unsigned char* edgeImg) {
    int lowstValue = 100; // حداقل value لبه 
    int min_size = 10;   // حداقل اندازه ضلع مربع
    /*
    ---> ------------
        |            |
        |            |
        |            |
        |            |
        --------------
    */ 
    // ما ابتدا این نقطه از مربع رو پیدا میکنیم
    // پیمایش در عکس
    for (int y = 1; y < height - min_size; y++) {
        for (int x = 1; x < width - min_size; x++) {
            
            // اگر یک پیکسل روشن نقطه شروع را اینجای مربع رو پیدا کردیم
            if (edgeImg[y * width + x] > lowstValue) {
                
                //  طول این خط عمودی را محاسبه می‌کنیم
                int len = 0;
                while ((y + len) < height - 1 &&  //از مرز عبور نکند
                    edgeImg[(y + len) * width + x] > lowstValue ) {
                    len++;
                }

                // اگر طول خط پیدا شده به اندازه کافی بزرگ بود
                if (len >= min_size) {
                    int right_x = x + len; 
                    /*
                         ------------ <----
                        |            |
                        |            |
                        |            |
                        |            |
                        --------------
                    */ 
                    int bottom_y = y + len;
                    /*
                         ------------ 
                        |            |
                        |            |
                        |            |
                        |            |
                        --------------
                        ^
                        |
                        |
                    */
                    // مطمئن می‌شویم از کادر تصویر خارج نشده باشیم
                    if (right_x < width - 2 && bottom_y < height - 2) {
                        int rightEdge = 0, topEdge = 0, bottomEdge = 0;
                        
                        for (int i = 0; i < len; i++) {
                            // بررسی لبه سمت راست 
                            //به طور کل با این روش حفظ میکند
                            if (edgeImg[(y + i) * width + right_x - 1] > lowstValue ||
                                edgeImg[(y + i) * width + right_x] > lowstValue ||
                                edgeImg[(y + i) * width + right_x + 1] > lowstValue) {
                                rightEdge++;
                            }
                            // بررسی لبه بالا
                            if (edgeImg[(y - 1) * width + x + i] > lowstValue ||
                                edgeImg[y * width + x + i] > lowstValue ||
                                edgeImg[(y + 1) * width + x + i] > lowstValue) {
                                topEdge++;
                            }
                            // بررسی لبه پایینی
                            if (edgeImg[(bottom_y - 1) * width + x + i] > lowstValue ||
                                edgeImg[bottom_y * width + x + i] > lowstValue ||
                                edgeImg[(bottom_y + 1) * width + x + i] > lowstValue) {
                                bottomEdge++;
                            }
                        }
                        
                        // اگر هر ۴ ضلع حداقل ۸۰ درصد در تصویر وجود داشتند، این شکل یک مربع است
                        if (rightEdge > len * 0.8 && topEdge > len * 0.8 && bottomEdge > len * 0.8) {
                            return 1; // مربع را پیدا کردیم
    
                        }
                    }
                }
            }
        }
    }
    return 0;
}

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
                _mm256_storeu_ps(results, sum); // ذخیره نتیجه در یک آرایه

                for(int k=0; k<8; k++) {
                    int val = (int)results[k];
                    
                    // محاسبه قدر مطلق برای لبه‌های منفی
                    if(val < 0) val = -val;
                    
                    //جلوگیری از سرریز
                    if(val > 255) val = 255;
                    
                    // ذخیره در خروجی
                    outputImg[(i*width + j+k)*channels + c] = (unsigned char)val; 
                } // ذخیره ی نتیجه در خروجی و کست کردن ان با کارکتر از اعشاری
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
                    int val = (int)sum;
                    if(val < 0) val = -val;
                    if(val > 255) val = 255;

                    outputImg[(i * width + j) * channels + c] = (unsigned char)val;
                                                                        //قرار دادن در ایندکس متناظر خروجی
                }
            }
        }


    return outputImg;
}
int main(){
    int width, height, channels;//تعداد کانال‌های رنگی (مثلاً 3 برای RGB و 4 برای RGBA)
    double total_c = 0.0; // متغیر محاسبه زمان تایع سی
    double total_asm = 0.0; //متغیر محاسبه زمان تابع اسمبلی
    int num_images = 100;
    
    // دو شمارنده مجزا برای مقایسه عملکرد تشخیص
    int imagesWithLineC = 0;
    int imagesWithLineASM = 0;


    float kernel[25] = { // فیلتر لاپلاسین (تشخیص لبه در تمام جهات)
        0,  0, -1,  0,  0,
        0, -1, -2, -1,  0,
       -1, -2, 16, -2, -1,
        0, -1, -2, -1,  0,
        0,  0, -1,  0,  0
    };

    printf("Processing %d images (Detection & Speed Comparison Mode)...\n\n", num_images);

    for (int n = 1; n <= num_images; n++) {// حلقه برای محاسبه هر عکس بصورت جداگانه

        char filename[64];
        sprintf(filename, "test_images/image%d.png",n); 
    

        unsigned char *img = stbi_load(filename, &width, &height, &channels, 3); 
        if(img == NULL){
            printf("Failed to load %s (Skipping)\n", filename);
            continue;
        }
        channels = 3;

        // ابتدا تبدیل به سیاه‌سفید (Grayscale)
        unsigned char* grayImg = calloc(width * height, 1); 
        for(int i = 0; i < width * height; i++) {
            int idx = i * channels;
            unsigned char R = img[idx];
            unsigned char G = img[idx + 1];
            unsigned char B = img[idx + 2];
            grayImg[i] = (unsigned char)(0.299f * R + 0.587f * G + 0.114f * B); 
        }

        // ۲. حذف نویز با کرنل گاوسی روی تصویر سیاه‌سفید
        float gaussian_kernel[25] = {
            1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
            4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
            1/273.0f,  4/273.0f,  7/273.0f,  4/273.0f, 1/273.0f
        };
        
        // به یک بافر موقت برای تصویر تار شده نیاز داریم
        // unsigned char* blurredImg = calloc(width * height, 1);
        // convolution2DInC(width, height, 1, grayImg, blurredImg, gaussian_kernel); 

        // تبدیل به float (اگر برای ASM نیاز داری)
        float *imgFloat = (float*)calloc(width * height, sizeof(float));
        for(int i = 0; i < width * height; i++)
            imgFloat[i] = (float)grayImg[i]; // <--- روی تصویر تار شده اعمال شد

        unsigned char *outputImage = calloc(width * height, 1);
        clock_t start, end;

        printf("\t--- Results for %s ---\n", filename);

        // تست سی
        start = clock();
        convolution2DInC(width, height, 1, grayImg, outputImage, kernel); 
        // تشخیص مربع
        int foundC = detectSquare(width, height, 1, outputImage); 

        end = clock();
        
        double time_c = (double)(end - start) / CLOCKS_PER_SEC; 
        total_c += time_c;
        if(foundC) imagesWithLineC++;

        // پاکسازی برای تست اسمبلی
        memset(outputImage, 0, width * height); 

        // تست asm
        start = clock();
        convolution2DInASM(width, height, 1, imgFloat, outputImage, kernel);
        int foundASM = detectSquare(width, height, 1, outputImage); 
        end = clock();

        double time_asm = (double)(end - start) / CLOCKS_PER_SEC; 
        total_asm += time_asm;
        if(foundASM) imagesWithLineASM++;

        // ذخیره نمونه خروجی فقط برای آخرین تصویر
        if (n == num_images) {
            stbi_write_png("final_result.png", width, height, 1, outputImage, width);
        }

        // آزادسازی حافظه‌های رزرو شده
        stbi_image_free(img);
        free(grayImg);
        // free(blurredImg); // <--- این متغیر جدید هم باید آزاد شود
        free(imgFloat);
        free(outputImage);
    }

    // گزارش کار
    printf("\t---  Report ---\n");
    printf("\tTotal Time C:   %f seconds\n", total_c);
    printf("\tTotal Time ASM: %f seconds\n", total_asm);
    printf("\t---  Square Detection ---\n");
    printf("\tTotal detection squares by C:   %d / %d images\n", imagesWithLineC, num_images);
    printf("\tTotal detections squares by ASM: %d / %d images\n", imagesWithLineASM, num_images);
    printf("\t---  Accuracy and speed's Report ---\n");
    printf("\tAccuracy : %d/%d\n",imagesWithLineC,70);//محاسبه دقت
    printf("\tASM is %f faster than c.\n",total_c/total_asm);
    printf("\t========================================\n");

    return 0;
}

