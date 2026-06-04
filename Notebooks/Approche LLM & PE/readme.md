# Extraction d'Informations Médicales et Résumé Clinique via LLM (Llama 3.1 8B)

## Vue d'ensemble du projet
Ce projet implémente un pipeline complet de traitement du langage naturel (NLP) appliqué au domaine médical. L'objectif est double : extraire des données structurées (dossier patient, démographie, données cliniques) à partir de notes médicales brutes, et générer un résumé clinique ciblé répondant à une question spécifique. 

L'approche (LLM & Prompt Engineering) repose exclusivement sur l'utilisation d'un Grand Modèle de Langage (LLM) optimisé et sur des techniques avancées de Prompt Engineering, garantissant une sortie structurée au format JSON pour une intégration directe dans des bases de données de santé (EHR).

## Architecture Technique et Choix d'Ingénierie

### 1. Sélection et Optimisation du Modèle
* **Modèle sélectionné :** `Meta-Llama-3.1-8B-Instruct`
* **Format et Quantification :** Utilisation du format GGUF (`Q4_K_M`, 4-bit quantization). Ce choix est justifié par la nécessité d'exécuter le modèle sur un environnement contraint en mémoire (Kaggle, GPU NVIDIA T4 avec 16 Go de VRAM). La quantification permet de diviser l'empreinte mémoire par quatre tout en conservant des performances quasi-identiques au modèle en précision FP16.
* **Moteur d'inférence :** `llama-cpp-python` compilé avec le support CUDA (`-DGGML_CUDA=on` et architecture `sm_75` spécifique au T4). Cela permet de décharger l'intégralité des couches du modèle (`n_gpu_layers=-1`) sur le GPU, accélérant drastiquement la vitesse d'inférence.

### 2. Stratégie de Prompt Engineering
Pour éviter les hallucinations et garantir l'exploitabilité des données, le prompt a été conçu avec des directives strictes :
* **Formatage contraint (JSON-only) :** Utilisation de l'argument `response_format={"type": "json_object"}` couplé à un prompt explicitant le schéma JSON attendu.
* **Règle du "Non mentionné" :** Instruction stricte forçant le modèle à utiliser des chaînes spécifiques ("Non mentionné") ou des listes vides (`[]`) si l'information n'est pas explicitement présente dans la note. Cette règle empêche le modèle d'inventer des diagnostics ou des traitements (zéro hallucination structurelle).
* **Séparation des temporalités :** Le schéma JSON divise les données cliniques (médicaments passés, traitements actuels, plan futur) pour faciliter l'analyse post-traitement.

## Méthodologie et Pipeline de Traitement

Le pipeline de traitement a été conçu pour être hautement résilient face aux instabilités des environnements cloud gratuits. 

**Note sur les données traitées :** Le jeu de données original (Ground Truth) contient 20 000 dossiers médicaux. Cependant, en raison des limites strictes de temps de calcul (timeouts) et des quotas d'heures GPU imposés par la plateforme Kaggle, cette première phase du projet a été exécutée sur un échantillon représentatif de 1 500 dossiers. Cet échantillon permet de valider de bout en bout l'architecture technique, d'analyser la qualité des résultats et de confirmer la viabilité de l'approche avant un passage à l'échelle.

Étapes clés du pipeline :
1.  **Gestion de la fenêtre de contexte :** Définition d'un `n_ctx=4096`. Les notes cliniques extrêmement longues sont tronquées aux 3000 premiers caractères pour garantir que le prompt et la réponse (jusqu'à 800 tokens) ne dépassent pas la limite du modèle.
2.  **Contrôle de la créativité :** La température est réglée très bas (`0.1`) pour forcer des réponses factuelles et déterministes.
3.  **Résilience et Checkpointing :** Implémentation d'une sauvegarde incrémentale toutes les 100 itérations. Si l'environnement de calcul s'interrompt, le script reprend automatiquement là où il s'est arrêté.
4.  **Gestion des erreurs :** Des blocs `try/except` gèrent les erreurs de mémoire (OOM) ou de décodage JSON en insérant un code d'erreur spécifique sans interrompre la boucle principale.

## Analyse des Résultats et Performances

L'évaluation a été réalisée sur l'échantillon des 1 500 dossiers et porte à la fois sur la stabilité du formatage et sur la qualité sémantique des résumés générés.

### Fiabilité de l'Extraction (Parsing JSON)
* **Taux de succès du formatage :** 100.0% (1500/1500 JSON valides).
* **Interprétation :** Le couplage du Prompt Engineering avec le grammar sampling de `llama.cpp` a fonctionné parfaitement. Aucune ligne n'a nécessité de retraitement manuel, ce qui valide la fiabilité du pipeline pour un usage en production.

### Évaluation Sémantique (Métriques NLP)
Les résumés générés ont été comparés aux réponses de référence (Ground Truth) en utilisant deux approches complémentaires :

| Métrique | Score Moyen | Description et Interprétation |
| :--- | :--- | :--- |
| **ROUGE-1** | 0.5648 | Mesure le chevauchement des mots (unigrammes). Un score supérieur à 0.5 en domaine médical indique une excellente récupération du vocabulaire clé. |
| **ROUGE-2** | 0.3891 | Mesure le chevauchement des paires de mots (bigrammes). Démontre que les expressions médicales composées sont souvent correctement reproduites. |
| **ROUGE-L** | 0.4695 | Mesure la plus longue sous-séquence commune. Confirme que la structure globale de la phrase de référence est respectée. |
| **BERTScore-F1** | 0.8900 | (Modèle : `distilbert-base-uncased`). Métrique basée sur la similarité sémantique des embeddings. Ce score extrêmement élevé (min: 0.74, max: 1.0) prouve que même lorsque le modèle reformule, le sens médical reste strictement identique à la référence. |

### Efficacité de la Synthèse (Compression)
* **Longueur moyenne des notes d'origine :** 279 mots
* **Longueur moyenne des résumés générés :** 48 mots
* **Taux de compression moyen :** 18.29%
* **Interprétation :** Le modèle réussit à condenser l'information d'un facteur 5, en ne gardant que l'essence clinique nécessaire pour répondre à la question posée, ce qui valide son utilité pour faire gagner du temps aux praticiens.

## Limites et Perspectives d'Amélioration

Bien que les résultats soient hautement significatifs sur cet échantillon, certaines évolutions sont prévues :
1.  **Passage à l'échelle (Scale-up) sur les 20 000 dossiers :** Le pipeline ayant prouvé sa robustesse à 100% de taux de parsing, l'objectif est désormais de traiter l'intégralité du dataset original. Cela nécessitera de migrer hors de Kaggle vers des instances cloud dédiées (AWS EC2, RunPod) pour s'affranchir des limites de temps.
2.  **Évaluation BERTScore spécialisée :** Le calcul du BERTScore a été réalisé avec un modèle généraliste (`distilbert-base-uncased`). L'utilisation d'un modèle d'embedding spécialisé comme `biobert-base-cased-v1.2` pourrait offrir une évaluation sémantique encore plus fine des terminologies médicales.
3.  **Gestion des notes très longues :** Actuellement, les notes sont tronquées brutalement à 3000 caractères. Pour des dossiers complexes, une approche RAG (Retrieval-Augmented Generation) ou une méthode de résumé hiérarchique (Map-Reduce) permettrait d'analyser l'intégralité d'un dossier sans perdre d'informations.
4.  **Évaluation humaine :** Bien que ROUGE et BERTScore soient élevés, une validation par un échantillon d'experts médicaux reste nécessaire pour évaluer la pertinence clinique exacte des informations extraites dans les listes de traitements et d'analyses.

## Guide d'Exécution

Pour reproduire ces résultats :
1. Importer le dataset dans un notebook configuré avec un accélérateur GPU T4.
2. Exécuter séquentiellement les cellules du script Python. L'installation initiale compilera la librairie avec CUDA.
3. Le modèle GGUF sera téléchargé automatiquement au premier lancement.
4. Les résultats intermédiaires et finaux seront générés dans le répertoire de travail local.