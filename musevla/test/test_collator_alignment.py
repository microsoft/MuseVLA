"""
Test script to verify PaddedCollatorForVLM label alignment is correct.
"""
import torch
from musevla.datasets.materialize import PaddedCollatorForVLM

def test_collator_alignment():
    """Test that labels are correctly aligned with prompt positions."""
    
    # Create mock collator
    collator = PaddedCollatorForVLM(
        model_max_length=512,
        pad_token_id=0,
        padding_side="right",
        extra_token_count=1,
    )
    
    # Create mock batch with known values
    # Sample 1: prompt_len=10, labels=[101, 102, 103] (3 tokens)
    # Sample 2: prompt_len=15, labels=[201, 202] (2 tokens)
    # Sample 3: prompt_len=8, labels=[301, 302, 303, 304] (4 tokens)
    
    batch = [
        {
            "pixel_values": torch.randn(3, 224, 224),
            "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),  # length 10
            "labels": torch.tensor([101, 102, 103]),  # length 3
            "prompt_len": 10,
            "dataset_name": "test1",
            "sensor_token": "none",
        },
        {
            "pixel_values": torch.randn(3, 224, 224),
            "input_ids": torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),  # length 15
            "labels": torch.tensor([201, 202]),  # length 2
            "prompt_len": 15,
            "dataset_name": "test2",
            "sensor_token": "thermal",
        },
        {
            "pixel_values": torch.randn(3, 224, 224),
            "input_ids": torch.tensor([31, 32, 33, 34, 35, 36, 37, 38]),  # length 8
            "labels": torch.tensor([301, 302, 303, 304]),  # length 4
            "prompt_len": 8,
            "dataset_name": "test3",
            "sensor_token": "radar",
        },
    ]
    
    # Run through collator
    print("=" * 80)
    print("Testing PaddedCollatorForVLM Label Alignment")
    print("=" * 80)
    
    result = collator(batch)
    
    print(f"\n[Input Batch Info]")
    for i, item in enumerate(batch):
        print(f"  Sample {i}: prompt_len={item['prompt_len']}, "
              f"input_ids_len={item['input_ids'].numel()}, "
              f"labels={item['labels'].tolist()}")
    
    print(f"\n[Collator Output Shapes]")
    print(f"  input_ids: {result['input_ids'].shape}")
    print(f"  attention_mask: {result['attention_mask'].shape}")
    print(f"  labels: {result['labels'].shape}")
    
    # Verify alignment for each sample
    print(f"\n[Label Alignment Verification]")
    max_input_len = result['input_ids'].shape[1]
    
    all_correct = True
    for i, item in enumerate(batch):
        prompt_len = item['prompt_len']
        expected_labels = item['labels']
        
        # Extract labels from result (skip -100 padding)
        result_labels = result['labels'][i]
        non_ignore_mask = result_labels != -100
        actual_labels = result_labels[non_ignore_mask]
        
        # Find where labels start (should be at prompt_len)
        label_positions = torch.where(non_ignore_mask)[0]
        if len(label_positions) > 0:
            label_start = label_positions[0].item()
            label_end = label_positions[-1].item() + 1
        else:
            label_start = -1
            label_end = -1
        
        # Check if alignment is correct
        expected_start = prompt_len  # Labels should start right after prompt
        is_correct = (label_start == expected_start) and torch.equal(actual_labels, expected_labels)
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"\n  Sample {i}: {status}")
        print(f"    Expected: labels at positions [{expected_start}:{expected_start + len(expected_labels)}] = {expected_labels.tolist()}")
        print(f"    Actual:   labels at positions [{label_start}:{label_end}] = {actual_labels.tolist()}")
        print(f"    Prompt length: {prompt_len}")
        print(f"    Labels shape in batch: {result['labels'][i].shape}")
        print(f"    Number of -100: {(result['labels'][i] == -100).sum().item()}")
        print(f"    Number of valid labels: {(result['labels'][i] != -100).sum().item()}")
        
        if not is_correct:
            all_correct = False
            # Show more details for failed cases
            print(f"    Full labels tensor (first 30): {result['labels'][i][:30].tolist()}")
            print(f"    Full labels tensor (around label pos): {result['labels'][i][max(0, label_start-5):label_end+5].tolist()}")
    
    print(f"\n{'=' * 80}")
    if all_correct:
        print("✓ ALL TESTS PASSED - Label alignment is correct!")
    else:
        print("✗ SOME TESTS FAILED - Label alignment has issues!")
    print("=" * 80)
    
    return all_correct

if __name__ == "__main__":
    success = test_collator_alignment()
    exit(0 if success else 1)
