## Travail réalisé
- Développement d'un CNN 2D léger sous PyTorch (1 282 paramètres).
- Test de l'architecture effectué avec succès.
- Chargement des 3 240 enregistrements cardiaques PhysioNet 2016.
- Génération des spectrogrammes Log-Mel.
- Test du CNN sur un véritable spectrogramme cardiaque.
- Analyse du déséquilibre des classes : 2 575 normaux et 665 anormaux.

## Travail en cours
- Préparation du script d'entraînement.
- Séparation des données d'entraînement et de validation.

## Prochaines étapes
- Entraîner le CNN 2D.
- Évaluer la sensibilité, la spécificité et la MAcc.
- Mesurer la taille mémoire et la latence.
- Expérimenter la quantification et le pruning.