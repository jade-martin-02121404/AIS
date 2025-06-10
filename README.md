# AIS
Developing a semi-automated workflow for the digital design of braces for Adolescent Idiopathic Scoliosis (AIS)

# Semi-Automated AIS Brace Design Pipeline

**Version 9**

A Python pipeline for quantifying and predicting 3D brace rectifications in Adolescent Idiopathic Scoliosis (AIS), based on paired pre- and post-rectification 3D scans and moulds.  
This repository implements:

1. **Anatomical Landmark Analysis**  
   - Read Excel landmarks for each subject  
   - Compute ΔX/ΔY/ΔZ displacements (boxplots: Fig 1)  
2. **Centroid Trajectories & Surface Deviations**  
   - Align meshes to pelvis (ASIS)  
   - Slice cross-sections → centroid shifts (Fig 2)  
   - Rigid + nonrigid registration → max‐deviation maps (Fig 3)  
3. **Statistical Shape Modelling & PCA**  
   - Build pre, post, and combined PCA‐based SSM  
   - Auto‐select components to explain ≥ 95 % variance  
   - Leave-one-out reconstruction error (Fig 4)  
4. **2D Deviation Maps → HOG → PCA → K-Means**  
   - Project coronal slab → 2D deviation image  
   - Segment connected regions → extract HOG, location, size features  
   - Dimensionality reduction + K-Means → discrete rectification zones  
   - Compare clusters vs. Cobb‐angle improvement labels  

## Requirements

- Python 3.8+  
- `numpy`, `pandas`, `matplotlib`, `seaborn`  
- `scikit-learn`, `scikit-image`  
- `open3d`, `vtk`, `PyQt5`  
- [Ampscan](https://github.com/…/ampscan) (on your `PYTHONPATH`)  

Install via:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image open3d vtk PyQt5
# plus your local ampscan package

