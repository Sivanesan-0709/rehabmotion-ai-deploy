# External Rehabilitation Research Datasets

This directory stores external clinical and biomechanical datasets used for model training, subject-independent validation (LOSO), and Sim-to-Real domain transfer experiments.

## 1. UI-PRMD (University of Idaho Physical Rehabilitation Movement Dataset)
- **Source:** [https://webpages.uidaho.edu/ui-prmd/](https://webpages.uidaho.edu/ui-prmd/)
- **Citation:** Vakanski et al., *A Physical Rehabilitation Movement Dataset with Kinematic Measurements*, Nature Scientific Data, 2018.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Directory:** `data/external_datasets/ui_prmd/raw/`
- **Supported Exercises:** Shoulder Abduction, Shoulder Flexion, Elbow Flexion, Knee Extension, Sit-to-Stand, Deep Squat, Hurdle Step, Inline Lunge, Side Lunge, Trunk Rotation.

## 2. KiMoRe (Kinematic Assessment of Movement in Rehabilitation)
- **Source:** [https://u-pad.unimc.it/handle/11393/301719](https://u-pad.unimc.it/handle/11393/301719)
- **Citation:** Capecci et al., *The KiMoRe dataset: Kinematic assessment of movement in rehabilitation*, IEEE Data in Brief, 2019.
- **License:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)
- **Directory:** `data/external_datasets/kimore/raw/`
- **Supported Exercises:** Lifting of Arms (Shoulder Abduction), Lateral Trunk Flexion, Trunk Rotation, Pelvis Rotations, Squatting.

## Namespace Isolation & Schema Integrity
All ingested samples are strictly namespaced (`UIPRMD_S01`..., `KIMORE_S01`...) to prevent ID collisions with local real recordings (`S001`...) and synthetic datasets (`subject_01`...). All features are validated against the unified 68-feature kinematic schema.
