# Unified Backend Services

This repository contains the consolidated backend infrastructure for various applications developed by Sean Fontaine. It utilizes a single deployment environment (FastAPI on Fly.io) to manage API services across multiple projects.

## Contained Services

This backend currently supports the following applications:

1.  **[800m Training & Race Calculator](https://www.seanfontaine.dev/en/800m-calculator)**
    *   Provides the core calculation and data processing services for 800m training and race prediction.

2.  **[Pokémon VGC Teammate Predictor](https://www.seanfontaine.dev/poke-team-predictor)**
    *   Hosts the prediction logic and data services for the Pokémon VGC Teammate Prediction App.

3.  **[Podcast Project – News vs Podcasts Analysis](https://www.seanfontaine.dev/podcast-project)**
    *   Manages the data and analysis endpoints related to the "Podcasting the News" project.

4.  **Little Scholars WhatsApp AI Assistant**
    *   Provides parents at Little Scholars (Hong Kong) with fast, multilingual answers over WhatsApp about courses, class hours, policies, tuition, assignments, and admissions, using retrieval-augmented generation powered by Amazon Bedrock and a Markdown knowledge base.

## Structure

The project is structured to allow multiple, independent services to run efficiently within a unified FastAPI application, optimizing deployment and resource usage.