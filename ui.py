import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="Reaper Fruit Scanner", page_icon="🍎")
st.title("🍎 نظام فحص جودة الفواكه والخضروات")
st.write("قم برفع صورة ليقوم الذكاء الاصطناعي بتحديد النوع والجودة فوراً.")

# 2. تحميل الموديل (تم تعديله ليتناسب مع الملفات المرفوعة)
@st.cache_resource
def load_my_model():
    # هيحاول يحمل الموديل بالاسم اللي ظاهر في الـ VS Code عندك
    model_path = 'fruit_quality_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # لو مش موجود هيحاول يحمل أي ملف موديل تاني
        return tf.keras.models.load_model('model.h5')

model = load_my_model()

# 3. الحصول على أسماء الفئات (تم كتابتها يدوياً لتعمل على السيرفر)
# هذه الأسماء مأخوذة من مجلد التدريب الخاص بك
class_names = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 
    'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 
    'freshtomato', 'rottentomato', 'rottenpotato', 'rottenoranges', 
    'rottenokra', 'rottencucumber', 'rottencapsicum', 'rottenbittergroud', 
    'rottenbanana', 'rottenapples'
]

# 4. واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة فاكهة أو خضار...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    img = Image.open(uploaded_file)
    st.image(img, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري التحليل...'):
        # 5. معالجة الصورة لتناسب الموديل (Size 224x224)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. التنبؤ
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        
        # التأكد من أن الـ Index لا يتخطى عدد الفئات
        if class_idx < len(class_names):
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
        else:
            st.error("خطأ: الموديل يعطي نتائج خارج نطاق الفئات المعرفة.")