from dataclasses import dataclass, field
import time
import torch


@dataclass
class SequenceQoSState:
    seq_id: int
    start_time: float
    deadline: float
    remaining_tokens: int
    waiting_time: float
    phase: float


@dataclass
class QoSRuntimeContext:
    seq_ids: torch.Tensor
    seq_priority: torch.Tensor
    remaining_tokens: torch.Tensor
    phase_tensor: torch.Tensor

    flat_seq_priority: torch.Tensor
    flat_remaining_tokens: torch.Tensor
    flat_valid_mask: torch.Tensor

    last_hidden_states: torch.Tensor | None = None
    created_at: float = field(default_factory=time.time)


class RequestStateTable:
    def __init__(self, args):
        self.args = args
        self.table: dict[int, SequenceQoSState] = {}
        self.next_seq_id = 0

    def _phase_value(self, phase: str) -> float:
        if phase == "decode":
            return self.args.qos_phase_decode
        return self.args.qos_phase_prefill

    def _build_deadline(self, now: float, remaining_tokens: int) -> float:
        return (
            now
            + self.args.qos_default_deadline_s
            + remaining_tokens * self.args.qos_avg_token_cost
        )

    def _sequence_priority(
        self,
        slack: torch.Tensor,
        waiting: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        urgency = 1.0 / slack
        return (
            self.args.qos_w1 * urgency
            + self.args.qos_w2 * waiting
            + self.args.qos_w3 * phase
        )

    def _get_or_create_state(
        self,
        seq_id: int,
        now: float,
        remaining_tokens: int,
        phase_value: float,
    ) -> SequenceQoSState:
        if seq_id not in self.table:
            self.table[seq_id] = SequenceQoSState(
                seq_id=seq_id,
                start_time=now,
                deadline=self._build_deadline(now, remaining_tokens),
                remaining_tokens=remaining_tokens,
                waiting_time=0.0,
                phase=phase_value,
            )
        else:
            st = self.table[seq_id]
            st.remaining_tokens = remaining_tokens
            st.phase = phase_value
            st.waiting_time = max(now - st.start_time, 0.0)

        return self.table[seq_id]

    def build_context(self, batch: dict, phase: str = "prefill") -> QoSRuntimeContext:
        now = time.time()

        attn = batch["attention_mask"]
        if not torch.is_tensor(attn):
            attn = torch.tensor(attn)
        attn = attn.detach()

        B, L = attn.shape
        device = attn.device

        remaining_tokens = attn.sum(dim=1).float()
        phase_value = self._phase_value(phase)

        if "_qos_seq_ids" in batch:
            raw_seq_ids = batch["_qos_seq_ids"]
            if not torch.is_tensor(raw_seq_ids):
                raw_seq_ids = torch.tensor(raw_seq_ids)
            seq_ids_cpu = raw_seq_ids.detach().cpu().tolist()
        else:
            seq_ids_cpu = list(range(self.next_seq_id, self.next_seq_id + B))
            self.next_seq_id += B

        waiting = []
        deadlines = []
        phase_tensor_list = []

        for i in range(B):
            seq_id = int(seq_ids_cpu[i])
            rem = int(remaining_tokens[i].item())

            st = self._get_or_create_state(
                seq_id=seq_id,
                now=now,
                remaining_tokens=rem,
                phase_value=phase_value,
            )

            waiting.append(st.waiting_time)
            deadlines.append(st.deadline)
            phase_tensor_list.append(st.phase)

        seq_ids = torch.tensor(seq_ids_cpu, device=device, dtype=torch.long)
        waiting = torch.tensor(waiting, device=device, dtype=torch.float)
        deadlines = torch.tensor(deadlines, device=device, dtype=torch.float)
        phase_tensor = torch.tensor(phase_tensor_list, device=device, dtype=torch.float)

        slack = torch.clamp(
            deadlines - now - remaining_tokens * self.args.qos_avg_token_cost,
            min=self.args.qos_epsilon,
        )

        seq_priority = self._sequence_priority(slack, waiting, phase_tensor)

        flat_seq_priority = seq_priority.repeat_interleave(L)
        flat_remaining_tokens = remaining_tokens.repeat_interleave(L)
        flat_valid_mask = attn.reshape(-1).bool()

        return QoSRuntimeContext(
            seq_ids=seq_ids,
            seq_priority=seq_priority,
            remaining_tokens=remaining_tokens,
            phase_tensor=phase_tensor,
            flat_seq_priority=flat_seq_priority,
            flat_remaining_tokens=flat_remaining_tokens,
            flat_valid_mask=flat_valid_mask,
        )

    def mark_finished(self, seq_ids):
        if torch.is_tensor(seq_ids):
            seq_ids = seq_ids.detach().cpu().tolist()

        for seq_id in seq_ids:
            seq_id = int(seq_id)
            if seq_id in self.table:
                del self.table[seq_id]