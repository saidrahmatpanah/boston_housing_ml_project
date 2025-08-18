# 🚀 راهنمای شروع سریع - پروژه ماشین لرنینگ خانه‌های بوستون

## 📋 پیش‌نیازها

- Python 3.7 یا بالاتر
- pip (مدیر بسته‌های Python)
- Git

## ⚡ نصب و راه‌اندازی سریع

### 1. کلون کردن پروژه
```bash
git clone https://github.com/yourusername/boston_housing_ml_project.git
cd boston_housing_ml_project
```

### 2. نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### 3. اجرای دمو
```bash
python demo.py
```

## 📖 راهنمای کامل

### 🎯 اجرای نوت‌بوک‌ها
```bash
jupyter notebook notebooks/
```

نوت‌بوک‌ها به ترتیب زیر اجرا می‌شوند:

1. **01_data_exploration.ipynb** - تحلیل اکتشافی داده‌ها
2. **02_data_preprocessing.ipynb** - پیش‌پردازش داده‌ها
3. **03_modeling.ipynb** - آموزش مدل‌ها
4. **04_evaluation.ipynb** - ارزیابی مدل‌ها

### 🤖 اجرای کامل پایتلاین
```bash
python src/main.py
```

### 🧪 اجرای تست‌ها
```bash
python tests/test_models.py
```

## 📊 ساختار پروژه

```
boston_housing_ml_project/
├── README.md                 # راهنمای اصلی پروژه
├── requirements.txt          # وابستگی‌های Python
├── setup.py                 # فایل نصب پروژه
├── config.py                # تنظیمات پروژه
├── demo.py                  # اسکریپت دمو
├── src/                     # کدهای اصلی
│   ├── data_loader.py      # بارگذاری داده‌ها
│   ├── preprocessing.py    # پیش‌پردازش
│   ├── models.py          # مدل‌های ماشین لرنینگ
│   ├── evaluation.py      # ارزیابی مدل‌ها
│   └── main.py            # پایتلاین اصلی
├── notebooks/              # نوت‌بوک‌های Jupyter
├── tests/                  # تست‌های پروژه
├── data/                   # داده‌ها
├── models/                 # مدل‌های ذخیره شده
└── results/                # نتایج و گزارش‌ها
```

## 🔧 تنظیمات

فایل `config.py` شامل تمام تنظیمات پروژه است:

- نوع مقیاس‌بندی (Standard, Robust, MinMax)
- روش شناسایی outliers
- هیپرپارامترهای مدل‌ها
- مسیرهای فایل‌ها
- آستانه‌های عملکرد

## 📈 ویژگی‌های کلیدی

- **10 مدل مختلف**: رگرسیون خطی، رندوم فارست، XGBoost و...
- **پیش‌پردازش کامل**: مدیریت outliers، مقیاس‌بندی، تقسیم داده‌ها
- **بهینه‌سازی خودکار**: تنظیم هیپرپارامترها با GridSearchCV
- **ارزیابی جامع**: معیارهای مختلف (R²، RMSE، MAE، MAPE)
- **تحلیل اهمیت ویژگی‌ها**: برای مدل‌های tree-based
- **نمودارهای تعاملی**: مقایسه مدل‌ها، تحلیل residuals
- **گزارش‌های خودکار**: خلاصه عملکرد و توصیه‌ها

## 🎯 نتایج مورد انتظار

پس از اجرای کامل پروژه:

- **بهترین مدل**: معمولاً Random Forest یا XGBoost با R² > 0.8
- **فایل‌های تولید شده**:
  - `models/best_model.pkl` - بهترین مدل
  - `results/evaluation_report.txt` - گزارش ارزیابی
  - `results/training_results.json` - نتایج آموزش
  - نمودارهای مختلف تحلیل

## 🚨 رفع مشکلات رایج

### خطای Import
```bash
pip install -r requirements.txt
```

### خطای Dataset
```bash
python -c "from sklearn.datasets import load_boston; print('Dataset available')"
```

### خطای Memory
کاهش تعداد مدل‌ها در `config.py` یا استفاده از `demo.py`

## 📚 منابع بیشتر

- [README.md](README.md) - راهنمای کامل پروژه
- [Jupyter Notebooks](notebooks/) - آموزش‌های تعاملی
- [Source Code](src/) - کدهای اصلی پروژه
- [Tests](tests/) - تست‌های پروژه

## 🤝 مشارکت

برای مشارکت در پروژه:

1. Fork کنید
2. شاخه جدید ایجاد کنید
3. تغییرات را commit کنید
4. Pull Request ارسال کنید

## 📞 پشتیبانی

- **Issues**: [GitHub Issues](https://github.com/yourusername/boston_housing_ml_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/boston_housing_ml_project/discussions)
- **Wiki**: [GitHub Wiki](https://github.com/yourusername/boston_housing_ml_project/wiki)

---

⭐ اگر این پروژه برایتان مفید بود، لطفاً آن را ستاره‌دار کنید!
