# Download stockmarket company perimeters in HuggingFace
huggingface-cli login
# API key needs write role assigned
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
HF_USERNAME=Chuyin980321 PYTHONPATH=$PYTHONPATH:. TWILIO_ACCOUNT_SID= TWILIO_AUTH_TOKEN= TWILIO_FROM= TWILIO_TO= \
python3 systematic_trading/scripts/download_datasets.py \
  --mode production \
  --slot after-close \
  --timeoffset 3

# Download SSRN paper abtract to a Kili project of id YOUR_KILI_PROJECT_ID
PYTHONPATH=$PYTHONPATH:. \
KILI_API_KEY="ed2b35e8-3cd7-470b-be39-643f8e2aabad" \
python3 systematic_trading/strategy_ideas \
  --mode abstract \
  --jel-code G14 \
  --from-page 1 \
  --kili-project-id clm61qa112oiq081o0wwcfgs4

:'
G10 - General
G11 - Portfolio Choice; Investment Decisions
G12 - Asset Pricing
G13 - Contingent Pricing; Futures Pricing
G14 - Information and Market Efficiency; Event Studies
G15 - International Financial Markets
G18 - Government Policy and Regulation
G19 - Other
'

# Use the abtracts labeled in YOUR_SOURCE_KILI_PROJECT_ID to download SSRN paper PDF
# into another Kili project YOUR_TARGET_KILI_PROJECT_ID
PYTHONPATH=$PYTHONPATH:. python3 systematic_trading/strategy_ideas \
  --mode paper \
  --src-kili-project-id clm61qa112oiq081o0wwcfgs4 \
  --tgt-kili-project-id clm5z9ud30rch082d5tzw4n7l

# Transform the annotations of YOUR_KILI_PROJECT_ID into markdown strategy ID cards
PYTHONPATH=$PYTHONPATH:. python3 systematic_trading/strategy_ideas \
  --mode summary \
  --kili-project-id [clm5z9ud30rch082d5tzw4n7l] \
  --tgt-folder [/home/chuyin/systematic-trading/systematic_trading/strategy_ideas/details]