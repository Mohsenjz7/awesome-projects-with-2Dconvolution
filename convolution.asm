
section .text
    global convolution2DInASM1
; rdi = width 
; rsi = height 
; rdx = channels
; rcx = img
; r8  =  outputImg
; r9  = kernel

convolution2DInASM1:
    ; ساخت پشته 
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
;    رزرو حافظه در استک
    sub rsp, 32

; __m256i INDEX = _mm256_setr_epi32(
;         0, channels, 2*channels, 3*channels, 
;         4*channels, 5*channels, 6*channels, 7*channels
;     );
; دستور پایین معادل دستور بالاست
    vmovdqa ymm7, [rel seq_0_7];
    vpbroadcastd ymm6, edx      
    vpmulld ymm8, ymm7, ymm6      ;ضرب اعداد صحبح در avx

    mov r10d, 2                   ; i = 2 

loop_i:
    mov eax, esi                  ; eax = height
    sub eax, 2                    ; height - 2
    cmp r10d, eax
    jge end_loop_i               ; خروج از حلقه اگه دو تا کمتر از ارتفاع


    mov r11d, 2                   ; j = 2

loop_j:
    mov eax, edi                  ; eax = width
    sub eax, 10                   ; width - 10
    cmp r11d, eax
    jge end_loop_j               ; خروج از حلقه اگه از عرض ۱۰ تا کمتر باشه

    xor r12d, r12d                ; c = 0

loop_c:
    cmp r12d, edx                 ; پیمایش در channels
    jge end_loop_c

    cmp r12d, 3 ;اگر A داشت اونو بنویسه و کپی کنه در خروجی و دیگه محاسبه نکنه
    jne processـchannel
    jmp next_c 

processـchannel:
    vxorps ymm0, ymm0, ymm0       ; sum = [0,0,0,0,0,0,0,0]
    xor r15d, r15d                ; شمارنده برای کرنل
    
    mov r13d, -2                  ; شروع پیمایش در خود کرنل 
                                    ; h=-2

loop_h:
    cmp r13d,2; 2 اگر h برابر ۲ بود پایان
    jg end_loop_h

    xor r14d, r14d                ; w = 0 
loop_w:
    cmp r14d, 5   ;اگر w برابر 5 بود پایان 
    jge end_loop_w

    
    ; base = ((i+h)*width + j-2) * channels + c
    mov eax, r10d                 ; eax = i
    add eax, r13d                 ; eax = i + h
    imul eax, edi                 ; eax = (i + h) * width
    add eax, r11d                 ; eax = ... + j
    sub eax, 2                    ; eax = ... + j - 2
    imul eax, edx                 ; eax = (...) * channels
    add eax, r12d                 ; eax = base

    ; float kernelDer = kernel[counter++];
    movsxd rbx, r15d              ; rbx = counter     move with sign extended
    vbroadcastss ymm1, dword [r9 + rbx*4] ; ymm1 = valuOfKernel ,value of kernel : [n,n,n,n,n,n,n,n]  
    inc r15d                      ; counter++

    ; vgatherdps ymm2, [img + base*4 + w*channels*4 + INDEX*4], mask
    mov ebx, r14d                 ; ebx = w
    imul ebx, edx                 ; ebx = w * channels
    add ebx, eax                  ; ebx = base + w * channels
    
   
    vpcmpeqd ymm3, ymm3, ymm3     ;برای اینکه لاین های رجبستر ymm3  رو ۱ کنه  
    movsxd rbx, ebx
    lea rbx, [rcx + rbx*4]        ; rbx = آدرس پایه برای این 8 پیکسل
    
    
    vgatherdps ymm2, [rbx + ymm8*4], ;ymm3 اجرای دستور gather , یعنی بیاد از نقاط پراکنده حافظه داده بخونه
    ; Address = rbx + (ymm8[i] * 4)

    
    vfmadd231ps ymm0, ymm2, ymm1  ; ymm0 = (ymm2 * ymm1) + ymm0

    inc r14d                      ; w++
    jmp loop_w
end_loop_w:

    inc r13d                      ; h++
    jmp loop_h
end_loop_h:

    ; ذخیره ی نتیجه در حافظه
    vmovups [rsp], ymm0
    

    ; خواندن داده در رجیستر بزرگ و قرار دادن هر عدد اعشاری در یک پیکسل
    xor rbx, rbx                  ; k = 0
loop_k:
    cmp rbx, 8
    jge end_loop_k

    ; تبدیل اعشار به عدد صحیح 
    cvttss2si eax, dword [rsp + rbx*4]
    
    ; محاسبه ایندکس ذخیره‌سازی: ((i*width + j+k)*channels + c)
    mov r14d, r10d
    imul r14d, edi
    add r14d, r11d
    add r14d, ebx                 ; + k
    imul r14d, edx                ; * channels
    add r14d, r12d                ; + c
    

    movsxd r14, r14d
    mov byte [r8 + r14], al       ; ذخیره کست شده بصورت unsigned char

    inc rbx
    jmp loop_k
end_loop_k:

next_c:
    inc r12d                      ; c++
    jmp loop_c
end_loop_c:

    add r11d, 8                   ; j += 8
    jmp loop_j
end_loop_j:

    inc r10d                      ; i++
    jmp loop_i
end_loop_i:

    ; ازادسازی حافظه و خروجی کردن مقدار عکس خروجی
    mov rax, r8                   
    add rsp, 32                  
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
section .note.GNU-stack noalloc noexec nowrite progbits
section .data
    align 32
    seq_0_7 dd 0, 1, 2, 3, 4, 5, 6, 7
