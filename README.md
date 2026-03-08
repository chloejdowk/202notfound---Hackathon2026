# 202notfound---Hackathon2026
This repository is created for NEUROhackathon2026

# Intro
Brain-computer interfaces (BCIs) offer a promising approach for translating neural activity into external control, with important applications in rehabilitation and assistive technology. In this project, we developed a low-cost EEG-based BCI system that combines consumer-grade hardware, deep learning, and interactive simulation. Our dataset included both self-collected EEG recordings using OpenBCI and publicly available motor imagery data from OpenNeuro, allowing us to prototype and evaluate the system across multiple data sources.

To focus on motor-related neural signals, we selected channels centered on C3, C4, and Cz and extracted epochs corresponding to left- and right-hand motor tasks. These EEG time-series segments were used to train a one-dimensional convolutional neural network (1D CNN) to classify left versus right motor activity directly from EEG signals. The trained model was then integrated with a PyBullet virtual environment, where predicted motor intent controlled the movement of a simulated object.

This project demonstrates a proof of concept for accessible BCI systems that could one day support patients with paralysis by enabling control of assistive devices, virtual prosthetics, or rehabilitation interfaces without relying on residual muscle movement. More broadly, our work highlights the potential of low-cost neural technologies to expand access to neurorehabilitation tools while also revealing key challenges in signal variability, generalization, and robust real-time implementation.

# Presentation
https://www.canva.com/design/DAHDUHe8USQ/gCGMwNxoBoB02GmEKF65cA/edit

# Data Source
https://openneuro.org/datasets/ds004362/versions/1.0.0/file-display/README
