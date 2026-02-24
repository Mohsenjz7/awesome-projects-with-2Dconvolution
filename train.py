import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# تعریف مدل CNN ساده
class SimpleCNN(nn.Module): #ساخت کلاس که از شبکه عصبی ارث بری میکند
    def __init__(self):
        super(SimpleCNN, self).__init__()# کدهای مقداردهی اولیه کلاس پدر اجرا شود.

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=2, bias=False)#یک لایه کانولوشن دو بعدی می‌سازد.
        # حاشیه تصویر را با صفر پر می‌کند تا ابعاد خروجی کانولوشن دقیقاً همان (سایز تصویر اصلی) باقی بماند

        self.fc = nn.Linear(28 * 28, 10)        # لایه متصل متراکم (Fully Connected) برای طبقه‌بندی به ۱۰ کلاس (اعداد ۰ تا ۹)

    # Forward Propagation
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x) # تابع فعال‌ساز
        x = x.view(x.size(0), -1) # تصویر 28x28 را به یک آرایه خطی 784 تایی تبدیل می‌کند.
        x = self.fc(x) #داده برگردانده میشه و نتیجه نهایی (10 عدد) برگردانده می‌شود.
        return x

# آماده‌سازی دیتاست MNIST
transform = transforms.Compose([transforms.ToTensor()]) #تبدیل کردن عکس‌های خام به تنسورهای پای‌تورچ و نرمال‌سازی آن‌ها بین صفر و یک.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)#دانلود دیتاست
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)#۶۴ تصویر به صورت همزمان وارد شبکه می‌شوند.

model = SimpleCNN() #فراخوانی مدلی که قبلا ساختیم
criterion = nn.CrossEntropyLoss()#تابع هزینه
optimizer = optim.Adam(model.parameters(), lr=0.01) #وزن‌های مدل را در جهت کاهش خطا آپدیت کند

# ۳. آموزش مدل (فقط ۱ اپوک برای تست کافیه تا دقت قابل قبولی بده)
print("Training Model...")
for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training Finished!")

# ۴. استخراج وزن‌ها و ذخیره به فرمت فایل هدر C (weights.h)
conv_kernel = model.conv1.weight.data.numpy().flatten()
fc_weight = model.fc.weight.data.numpy()
fc_bias = model.fc.bias.data.numpy().flatten()

with open("weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    
    # آرایه کرنل کانولوشن
    f.write("float conv_kernel[25] = {")
    f.write(", ".join([str(w) for w in conv_kernel]))
    f.write("};\n\n")
    
    # بایاس لایه آخر
    f.write("float fc_bias[10] = {")
    f.write(", ".join([str(b) for b in fc_bias]))
    f.write("};\n\n")
    
    # وزن‌های لایه آخر
    f.write("float fc_weight[10][784] = {\n")
    for row in fc_weight:
        f.write("  {" + ", ".join([str(w) for w in row]) + "},\n")
    f.write("};\n\n")
    
    f.write("#endif\n")

print("Weights exported to weights.h successfully!")