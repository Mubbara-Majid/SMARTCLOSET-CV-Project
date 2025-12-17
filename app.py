import streamlit as st
import cv2
import numpy as np
from analyzer import analyze_item
from recommender import recommend_outfit

# Page configuration
st.set_page_config(
    page_title="Smart Closet",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #E8E6F0; padding: 2rem 3rem; }
    [data-testid="stSidebar"] { background-color: #D8D6E4; padding: 0; }
    [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.5rem; }
    .sidebar-logo { font-size: 2rem; font-weight: 700; color: #1a1a1a; margin-bottom: 3rem; padding-left: 0.5rem; }
    
    .sidebar-button {
        background-color: transparent; border: none; border-radius: 20px;
        padding: 1rem 1.5rem; margin: 0.5rem 0; width: 100%;
        text-align: left; cursor: pointer; font-size: 1rem; color: #5B5B8D;
        display: flex; align-items: center; gap: 0.75rem; transition: all 0.3s;
    }
    .sidebar-button:hover { background-color: rgba(91, 91, 141, 0.1); }
    
    .stButton > button {
        background: linear-gradient(90deg, #6B5FD8 0%, #8B7FE8 100%);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 2rem; font-weight: 600; font-size: 1rem;
    }
    
    .main-title { font-size: 1.8rem; font-weight: 600; color: #1a1a1a; }
    .collection-header { font-size: 1.5rem; font-weight: 600; color: #1a1a1a; margin-bottom: 1rem; }
    .stSelectbox > div > div { background-color: #E0DDF0; border: none; border-radius: 12px; padding: 0.5rem; color: #5B5B8D; }
    
    .item-card {
        background-color: white; border-radius: 20px; padding: 1.5rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s; height: 100%;
    }
    .item-card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.12); }
    
    .item-image { width: 100%; height: 180px; object-fit: contain; margin-bottom: 1rem; border-radius: 12px; background-color: #F5F4F8; }
    .item-name { font-weight: 600; color: #1a1a1a; font-size: 1rem; margin-bottom: 0.25rem; }
    .item-category { color: #5B5B8D; font-size: 0.9rem; }
    
    [data-testid="stFileUploader"] { background-color: white; border-radius: 16px; padding: 2rem; border: 2px dashed #6B5FD8; }
    
    .outfit-container {
        background: linear-gradient(135deg, #6B5FD8 0%, #8B7FE8 100%);
        border-radius: 24px; padding: 3rem; margin-top: 3rem; color: white;
    }
    .outfit-title { font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 2rem; color: white; }
    
    .stat-card {
        background: linear-gradient(135deg, #6B5FD8 0%, #8B7FE8 100%);
        border-radius: 16px; padding: 2rem; text-align: center; color: white; margin-bottom: 1rem;
    }
    .stat-number { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .stat-label { font-size: 1rem; opacity: 0.9; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state: st.session_state.page = 'wardrobe'
if 'closet' not in st.session_state: st.session_state.closet = []

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Smart Closet</div>', unsafe_allow_html=True)
    if st.button("👔  My Wardrobe", key="nav_wardrobe", use_container_width=True, type="primary" if st.session_state.page == 'wardrobe' else "secondary"):
        st.session_state.page = 'wardrobe'
        st.rerun()
    if st.button("✨  Outfit Stylist (AI)", key="nav_stylist", use_container_width=True, type="primary" if st.session_state.page == 'stylist' else "secondary"):
        st.session_state.page = 'stylist'
        st.rerun()
    if st.button("📤  Upload New Item", key="nav_upload", use_container_width=True, type="primary" if st.session_state.page == 'upload' else "secondary"):
        st.session_state.page = 'upload'
        st.rerun()
    if st.button("📊  Wear Stats", key="nav_stats", use_container_width=True, type="primary" if st.session_state.page == 'stats' else "secondary"):
        st.session_state.page = 'stats'
        st.rerun()

# --- PAGE: WARDROBE ---
if st.session_state.page == 'wardrobe':
    col1, col2 = st.columns([3, 1])
    with col1: st.markdown('<div class="main-title">Welcome to Your Digital Wardrobe</div>', unsafe_allow_html=True)
    with col2:
        if st.button("+ Add New Item", key="add_btn"):
            st.session_state.page = 'upload'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="collection-header">My Current Collection</div>', unsafe_allow_html=True)
    
    # Filter (Added 'Dresses')
    filter_option = st.selectbox("Filter", ["All Items", "Tops", "Bottoms", "Dresses", "Shoes"], label_visibility="collapsed")
    
    if len(st.session_state.closet) > 0:
        filtered_items = st.session_state.closet
        if filter_option != "All Items":
            category_map = {"Tops": "top", "Bottoms": "bottom", "Shoes": "shoe", "Dresses": "body"}
            if filter_option in category_map:
                filtered_items = [i for i in st.session_state.closet if i["category"] == category_map[filter_option]]
        
        cols_per_row = 4
        for i in range(0, len(filtered_items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(filtered_items):
                    item = filtered_items[i + j]
                    with col:
                        st.markdown('<div class="item-card">', unsafe_allow_html=True)
                        image_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, use_column_width=True)
                        display_name = item['name'].replace('.jpg', '').replace('.png', '').replace('_', ' ').title()
                        st.markdown(f'<div class="item-name">{display_name}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="item-category">| {item["category"].title()}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div style="text-align: center; padding: 4rem; background: white; border-radius: 20px; margin-top: 2rem;"><h2 style="color: #5B5B8D;">Your wardrobe is empty</h2><p style="color: #9B9BB8;">Start by uploading some items!</p></div>""", unsafe_allow_html=True)

# --- PAGE: UPLOAD ---
elif st.session_state.page == 'upload':
    st.markdown('<div class="main-title">Upload New Item</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="background: white; padding: 3rem; border-radius: 20px; text-align: center;"><h2 style="color: #1a1a1a; margin-bottom: 1rem;">📸 Upload Your Wardrobe Items</h2><p style="color: #5B5B8D;">Upload clear photos of your clothing items</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        new_items = 0
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing {file.name}...")
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if not any(item['name'] == file.name for item in st.session_state.closet):
                analyzed = analyze_item(image, file.name)
                st.session_state.closet.append(analyzed)
                new_items += 1
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        if new_items > 0:
            st.success(f"✅ Successfully added {new_items} items!")
            st.markdown("### Preview")
            cols = st.columns(min(4, new_items))
            for idx, item in enumerate(st.session_state.closet[-new_items:][:4]):
                with cols[idx]:
                    image_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                    st.caption(f"{item['category'].upper()}")
            if st.button("← Back to Wardrobe", use_container_width=True):
                st.session_state.page = 'wardrobe'
                st.rerun()

# --- PAGE: STYLIST (UPDATED) ---
elif st.session_state.page == 'stylist':
    st.markdown('<div class="main-title">AI Outfit Stylist</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    tops = [i for i in st.session_state.closet if i["category"] == "top"]
    bottoms = [i for i in st.session_state.closet if i["category"] == "bottom"]
    shoes = [i for i in st.session_state.closet if i["category"] == "shoe"]
    bodies = [i for i in st.session_state.closet if i["category"] == "body"] # Dresses
    
    # Stats (4 Columns now)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(tops)}</div><div class="stat-label">👕 Tops</div></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(bottoms)}</div><div class="stat-label">👖 Bottoms</div></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(bodies)}</div><div class="stat-label">👗 Dresses</div></div>""", unsafe_allow_html=True)
    with col4: st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(shoes)}</div><div class="stat-label">👟 Shoes</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Logic: Can we make an outfit?
    can_make_3pc = (len(tops) >= 1 and len(bottoms) >= 1 and len(shoes) >= 1)
    can_make_2pc = (len(bodies) >= 1 and len(shoes) >= 1)
    
    if can_make_3pc or can_make_2pc:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Generate Perfect Outfit", use_container_width=True):
                with st.spinner("Creating your outfit..."):
                    outfit = recommend_outfit(st.session_state.closet)
                    if outfit:
                        st.session_state.current_outfit = outfit
                        st.balloons()
        
        if hasattr(st.session_state, 'current_outfit') and st.session_state.current_outfit:
            st.markdown('<div class="outfit-container"><div class="outfit-title">✨ Your Perfect Outfit</div></div>', unsafe_allow_html=True)
            
            outfit = st.session_state.current_outfit
            # Dynamic Columns based on outfit size (2 or 3 items)
            num_items = len(outfit)
            if num_items == 3:
                cols = st.columns(3)
            else:
                # Center the 2 items
                _, c1, c2, _ = st.columns([1, 2, 2, 1])
                cols = [c1, c2]

            for col, item in zip(cols, outfit):
                with col:
                    st.markdown('<div class="item-card">', unsafe_allow_html=True)
                    image_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, use_column_width=True)
                    st.markdown(f'<div class="item-name">{item["name"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="item-category">{item["category"].title()}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ You need either (Top + Bottom + Shoe) OR (Dress + Shoe) to generate an outfit!")

# --- PAGE: STATS ---
elif st.session_state.page == 'stats':
    st.markdown('<div class="main-title">Wardrobe Statistics</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(st.session_state.closet) > 0:
        col1, col2, col3, col4, col5 = st.columns(5) # 5 columns now
        tops = len([i for i in st.session_state.closet if i["category"] == "top"])
        bottoms = len([i for i in st.session_state.closet if i["category"] == "bottom"])
        shoes = len([i for i in st.session_state.closet if i["category"] == "shoe"])
        bodies = len([i for i in st.session_state.closet if i["category"] == "body"])
        
        with col1: st.markdown(f"""<div class="stat-card"><div class="stat-number">{len(st.session_state.closet)}</div><div class="stat-label">Total</div></div>""", unsafe_allow_html=True)
        with col2: st.markdown(f"""<div class="stat-card"><div class="stat-number">{tops}</div><div class="stat-label">Tops</div></div>""", unsafe_allow_html=True)
        with col3: st.markdown(f"""<div class="stat-card"><div class="stat-number">{bottoms}</div><div class="stat-label">Bottoms</div></div>""", unsafe_allow_html=True)
        with col4: st.markdown(f"""<div class="stat-card"><div class="stat-number">{bodies}</div><div class="stat-label">Dresses</div></div>""", unsafe_allow_html=True)
        with col5: st.markdown(f"""<div class="stat-card"><div class="stat-number">{shoes}</div><div class="stat-label">Shoes</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Simple combo calculation (T*B*S) + (D*S)
        possible_outfits = (tops * bottoms * shoes) + (bodies * shoes)
        st.markdown(f"""<div class="outfit-container"><h2 style="font-size: 3rem; margin-bottom: 0.5rem;">{possible_outfits}</h2><p style="font-size: 1.2rem;">Possible outfit combinations!</p></div>""", unsafe_allow_html=True)
    else:
        st.info("No items in wardrobe yet!")