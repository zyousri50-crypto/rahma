import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Reaper Fruit Scanner", page_icon="🍎")
st.title("🍎 نظام فحص جودة الفواكه والخضروات")
st.write("قم برفع صورة ليقوم الذكاء الاصطناعي بتحديد النوع والجودة فوراً.")

# 2. تحميل الموديل (تأكد من اسم الملف الصحيح)
@st.cache_resource
def load_my_model():
    # جرب تحميل الامتداد المتاح لديك
    try:
        return tf.keras.models.load_model('fruit_quality_model.h5')
    except:
        return tf.keras.models.load_model('model.h5') # أو اسم الملف الذي ظهر لك في الصورة

model = load_my_model()

# 3. الحصول على أسماء الفئات (تلقائياً من مجلد التدريب)
# استبدل هذا المسار بمسار مجلد الـ train عندك
train_path = r"C:\Users\Zakaria Yousri\Downloads\archive\dataset\Train"
class_names = sorted(os.listdir(train_path))

# 4. واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة فاكهة أو خضار...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    img = Image.open(uploaded_file)
    st.image(img, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # 5. معالجة الصورة لتناسب الموديل
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. التنبؤ
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(predictions)
        result_label = class_names[class_idx]
        confidence = np.max(predictions) * 100

        # 7. عرض النتيجة بشكل احترافي
        st.divider()
        if "fresh" in result_label.lower():
            st.success(f"✅ النتيجة: {result_label}")
            st.info(f"نسبة التأكد: {confidence:.2f}%")
            st.balloons()
        else:
            st.error(f"⚠️ النتيجة: {result_label}")
            st.warning("تنبيه: هذه الفاكهة قد تكون تالفة أو بها عيوب جودة.")
            st.info(f"نسبة التأكد: {confidence:.2f}%")