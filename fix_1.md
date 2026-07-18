# Fix 1: invalidate resident KV between scheduler steps

## Symptom

MTP decode starts with plausible tokens and then degrades into repeated or
unrelated text after rebasing the SFA backend onto the colleague decode manager.

## Root cause

The pre-rebase working tree reset every resident LRU row in
`prepare_scheduler_step()`. The rebased integration only waited for pending D2H
copies, so rejected MTP draft positions could be rewritten in the CPU pool while
the resident LRU still reported the same request/token pair as a hit. Attention
then consumed stale K/V from the previous draft step.

## Fix

- Drain graph-stream host callbacks before changing their CPU LRU buffers.
- Invalidate request identity and current resident slots every scheduler step.
- Preserve the colleague C++ LRU, graph callback, and onload interfaces.

## Verification

- CPU unit coverage checks eager D2H draining and LRU invalidation.
- CPU unit coverage checks graph-stream draining happens before invalidation.
- Service-level MTP accuracy remains a manual verification step.
