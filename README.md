# Unified Backend Services

This repository contains the consolidated backend infrastructure for various applications developed by Sean Fontaine. It utilizes a single deployment environment (FastAPI on Fly.io) to manage API services and webhook fulfillment for content management and complex logic across multiple frontends.

## Contained Services

This backend currently supports the following applications, providing dedicated APIs and logic for each:

1.  **[800m Training & Race Calculator](https://www.seanfontaine.dev/en/800m-calculator)**
    *   Provides the core calculation and data processing services 800m training and race prediction.

2.  **[Pokémon VGC Teammate Predictor](https://www.seanfontaine.dev/poke-team-predictor)**
    *   Hosts the prediction logic and data services necessary the Pokemon VGC Teammate Prediction App.

3.  **[Podcast Project – News vs Podcasts Analysis](https://www.seanfontaine.dev/podcast-project)**
    *   Manages the data and analysis endpoints related to the "Podcasting the News" project.

4.  **WhatsApp Agent**
    *   The dedicated Dialogflow webhook fulfillment agent responsible for handling user inquiries for a WhatsApp agent frontend.

## Structure

The project is structured to allow multiple, independent services to run efficiently within a unified FastAPI application, optimizing deployment and resource usage.