from poker_trainer.engine.hand import HoldemHand
from poker_trainer.engine.models import ActionType, Position
from poker_trainer.engine.replay import ReplayBundle


def play_passively(hand: HoldemHand) -> None:
    while not hand.is_complete:
        actor = hand.current_actor_id
        legal = hand.legal_actions(actor)
        hand.act(actor, ActionType.CALL if legal.to_call else ActionType.CHECK)


def test_random_seed_and_action_log_replay_are_identical(six_seats):
    original = HoldemHand(six_seats(), seed=20260827, hand_id="replay-1")
    play_passively(original)
    payload = ReplayBundle.from_hand(original).to_json()
    replayed = ReplayBundle.from_json(payload).replay()

    assert {
        player_id: tuple(map(str, player.hole_cards))
        for player_id, player in original.players.items()
    } == {
        player_id: tuple(map(str, player.hole_cards))
        for player_id, player in replayed.players.items()
    }
    assert list(map(str, original.board)) == list(map(str, replayed.board))
    assert [record.as_dict() for record in original.history] == [
        record.as_dict() for record in replayed.history
    ]
    assert original.result == replayed.result
    assert {
        player_id: player.stack for player_id, player in original.players.items()
    } == {
        player_id: player.stack for player_id, player in replayed.players.items()
    }


def test_replay_can_stop_at_any_voluntary_action_prefix(six_seats):
    hand = HoldemHand(six_seats(), seed=99)
    for _ in range(3):
        actor = hand.current_actor_id
        hand.act(actor, ActionType.CALL)
    bundle = ReplayBundle.from_hand(hand)
    prefix = bundle.replay(action_count=2)
    assert prefix.current_actor_id == "CO"
    assert prefix.sequence == 4  # 两条盲注 + 两个自愿动作


def test_different_fixed_seed_changes_deal(six_seats):
    first = HoldemHand(six_seats(), seed=1)
    second = HoldemHand(six_seats(), seed=2)
    first_cards = [str(card) for player in first.players.values() for card in player.hole_cards]
    second_cards = [str(card) for player in second.players.values() for card in player.hole_cards]
    assert first_cards != second_cards


def test_uncalled_all_in_excess_is_refunded_and_replayed_exactly(six_seats):
    hand = HoldemHand(
        six_seats({Position.UTG: 500, Position.HJ: 300}),
        seed=100,
        hole_overrides={"UTG": ["As", "Ad"], "HJ": ["Ks", "Kd"]},
        board_override=["2c", "3d", "7h", "8s", "9c"],
    )
    for player_id, action in (
        ("UTG", ActionType.ALL_IN),
        ("HJ", ActionType.ALL_IN),
        ("CO", ActionType.FOLD),
        ("BTN", ActionType.FOLD),
        ("SB", ActionType.FOLD),
        ("BB", ActionType.FOLD),
    ):
        hand.act(player_id, action)

    refund = next(record for record in hand.history if record.action == ActionType.REFUND)
    assert refund.player_id == "UTG"
    assert refund.paid == -200
    assert refund.pot_before == 860
    assert refund.pot_after == 660
    assert hand.player("UTG").total_commitment == 300
    assert hand.result.payouts["UTG"] == 660

    replayed = ReplayBundle.from_json(ReplayBundle.from_hand(hand).to_json()).replay()
    assert [record.as_dict() for record in replayed.history] == [
        record.as_dict() for record in hand.history
    ]
    assert replayed.result == hand.result
    replayed.assert_chip_conservation()

