import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd

# 1. إعدادات الصفحة
st.set_page_config(page_title="Fruit Quality Scanner", page_icon="🍎")
st.title("🍎 نظام فحص جودة الفواكه والخضروات")
st.write("قم برفع صورة ليقوم الذكاء الاصطناعي بتحليلها بدقة.")

# 2. تحميل الموديل
@st.cache_resource
def load_my_model():
    model_path = 'fruit_quality_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return tf.keras.models.load_model('model.h5')

model = load_my_model()

# 3. الفئات الـ 18 المعتمدة من مجلد التدريب الخاص بك
class_names = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 
    'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 
    'freshtomato', 'rottentomato', 'rottenpotato', 'rottenoranges', 
    'rottenokra', 'rottencucumber', 'rottencapsicum', 'rottenbittergroud', 
    'rottenbanana', 'rottenapples'
]

# 4. واجهة رفع الصور
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='الصورة المرفوعة', use_container_width=True)
    
    with st.spinner('جاري التحليل واستخراج الاحتمالات...'):
        # 5. معالجة الصورة (224x224)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 6. التنبؤ واستخراج أعلى احتمالات
        predictions = model.predict(img_array)[0]
        top_indices = np.argsort(predictions)[-3:][::-1] # جلب أعلى 3 احتمالات
        
        main_idx = top_indices[0]
        second_idx = top_indices[1]
        
        result_label = class_names[main_idx]
        confidence = predictions[main_idx] * 100
        second_confidence = predictions[second_idx] * 100

        # 7. عرض النتيجة الذكية
        st.divider()
        
        # كاشف الحيرة: لو الفرق بين المركز الأول والثاني أقل من 15%
        if (confidence - second_confidence) < 15:
            st.warning(f"🤔 الموديل محتار قليلاً بين '{result_label}' و '{class_names[second_idx]}'")
            st.info("نصيحة: جرب تصوير الثمرة من زاوية أوضح أو إضاءة أفضل.")
        
        if "fresh" in result_label.lower():
            st.success(f"✅ التوقع الأقرب: {result_label}")
        else:
            st.error(f"⚠️ التوقع الأقرب: {result_label}")
            st.warning("تنبيه: جودة الثمرة قد تكون منخفضة.")

        # عرض جدول الاحتمالات لزيادة الشفافية
        with st.expander("🔍 عرض تفاصيل التحليل كاملة"):
            df_probs = pd.DataFrame({
                'الصنف': [class_names[i] for i in top_indices],
                'نسبة الثقة': [f"{predictions[i]*100:.2f}%" for i in top_indices]
            })
            st.table(df_probs)

st.markdown("---")
st.caption("Developed by Zakaria Yousri - King Salman International University")