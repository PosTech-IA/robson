import torch
print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
# Source - https://stackoverflow.com/a
# Posted by jodag, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-02, License - CC BY-SA 4.0

torch.zeros(1).cuda()
