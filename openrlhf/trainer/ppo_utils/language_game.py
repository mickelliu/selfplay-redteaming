import time
from red_team.utils import convert_game_history_to_messages, cot_format_check_and_extract, get_cot_formatting_reward
from openrlhf.utils.remote_rm_utils import remote_rm_fn

class DialogueGameManager:
    def __init__(self, tokenizer, remote_rm_url, strategy, custom_configs=None):
        self.tokenizer = tokenizer
        self.remote_rm_url = remote_rm_url
        self.strategy = strategy
        self.custom_configs = custom_configs or {}
        self.max_turns = self.custom_configs.get("max_turns", 2)
        self.reward_type = self.custom_configs.get("reward_type", "general_sum")
        self.no_attacker_turn = self.custom_configs.get("no_attacker_turn", False)
        self.no_defender_turn = self.custom_configs.get("no_defender_turn", False)
        self.disable_hidden_cot = self.custom_configs.get("direct_chat_no_cot", False)
        
        # Select reward function based on reward type
        if "general_sum" in self.reward_type:
            from red_team.utils import get_redteaming_game_reward_general_sum
            self.get_redteaming_game_reward = get_redteaming_game_reward_general_sum
        elif "zero_sum" in self.reward_type:
            from red_team.utils import get_redteaming_game_reward_zero_sum
            self.get_redteaming_game_reward = get_redteaming_game_reward_zero_sum
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
    def initialize_games(self, prompts, completions, data_types):
        """Set up initial game states from prompts"""
        self.active_games = {}
        idx = 0
        for prompt, completion, data_type in zip(prompts, completions, data_types):
            if self.no_defender_turn and data_type in ["vanilla_benign", "vanilla_harmful"]:
                continue # skip vanilla prompts when we only train the attacker

            self.active_games[idx] = {
                "history": [], # actual conversation history
                "raw_history": [], # raw conversation history without parsing away the hidden cot
                "processed_output_history": [], # output history with additional information
                "prompts": prompt,
                "prompt_type": data_type,
                "completion": completion,
                "finished": False,
                "current_turn": 0
            }
            idx += 1
        
    def play_games(self, attacker_llm_generator, defender_llm_generator, **kwargs):
        """Play the game for multiple turns"""
        active_games = self.active_games
        
        for turn in range(self.max_turns):
            if not active_games:
                break
                
            player_role = "attacker" if turn % 2 == 0 else "defender"
            
            if turn % 2 == 0:
                self.strategy.print(f"🎮 Turn {turn}: 🚀 Generating attacks... 🔥")
                self._generate_attacker_turn(active_games, turn, attacker_llm_generator, **kwargs)
            else:
                self.strategy.print(f"🎮 Turn {turn}: 🛡️ Generating defenses... 🛡️")
                self._generate_defender_turn(active_games, turn, defender_llm_generator, **kwargs)

            # Mark games as finished if last turn
            if turn == self.max_turns - 1:
                for game in active_games.values():
                    game["finished"] = True
        
        return active_games
                    

    def _generate_attacker_turn(self, active_games, turn, llm_generator, **kwargs):
        """Generate a turn for the attacker"""
        batch_chat_messages = []
        labels = []
        game_to_postprocess = []
        
        for idx, game in active_games.items():
            if game["finished"]:
                continue # skip finished games
                
            game['current_turn'] = turn

            # Three cases:
            # 1. If we have both attackers and defenders, attacker will generate attacks and another half of the prompts will be vanilla datapoints
            # 2. If we only train the defender, then all of the attack prompts will be vanilla datapoints, and this is handled long before by the dataset_loader
            # 3. If we only train the attacker, we might want to skip the vanilla prompts (both benign and harmful), because they are useless and their interactions will not be used for training nor used for wandb logging. At the moment this is done by deleting all such games in the initialize_games function.
            
            if game["prompt_type"] in ["vanilla_benign", "vanilla_harmful"] or self.no_attacker_turn:
                # Directly use the vanilla (benign or harmful) prompt as attacker's message
                game["history"].append({
                    "game_role": "attacker",
                    "content": game["prompts"]
                })
                game["raw_history"].append({
                    "game_role": "attacker",
                    "content": game["prompts"]
                })
                game["processed_output_history"].append({
                    "game_role": "attacker",
                    "turn": turn,
                    "output": game["prompts"],
                    "game_states": {},
                })
                game['is_generated_attack'] = False
                continue # skip non-generatable turns
            else:
                # Generate attack for harmful prompts
                game['is_generated_attack'] = True

            chat_message = convert_game_history_to_messages(
                game["history"],
                player_role="attacker",
                prompt=game["prompts"],
                prompt_type=game["prompt_type"],
                custom_configs=self.custom_configs,
                tokenizer=self.tokenizer
            )
            
            batch_chat_messages.append(chat_message)
            labels.append(game["prompt_type"])
            game_to_postprocess.append((idx, game))
        
        assert len(batch_chat_messages) == len(game_to_postprocess), "batch_chat_messages and game_to_postprocess must have the same length"
            
        # Only generate if we have messages to process
        if not batch_chat_messages:
            return [], []
            
        llm_outputs = llm_generator(batch_chat_messages, labels, **kwargs)
        self._process_responses_and_game_states(game_to_postprocess, llm_outputs, "attacker", turn)
    
    def _generate_defender_turn(self, active_games, turn, llm_generator, **kwargs):
        """Generate a turn for the defender"""
        batch_chat_messages = []
        labels = []
        game_to_postprocess = []
        
        for idx, game in active_games.items():
            if game["finished"]:
                continue
                
            game['current_turn'] = turn
            
            # messages = convert_game_history_to_query(
            #     game["history"],
            #     player_role="defender",
            #     prompt=game["prompts"],
            #     custom_configs=self.custom_configs
            # )
            
            # chat_message = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            # )

            chat_message = convert_game_history_to_messages(
                game["history"],
                player_role="defender",
                prompt=game["prompts"],
                prompt_type=game["prompt_type"],
                custom_configs=self.custom_configs,
                tokenizer=self.tokenizer
            )
            
            batch_chat_messages.append(chat_message)
            labels.append(game["prompt_type"])
            game_to_postprocess.append((idx, game))
            
        # Only generate if we have messages to process
        if not batch_chat_messages:
            return [], []
            
        llm_outputs = llm_generator(batch_chat_messages, labels, **kwargs)
        self._process_responses_and_game_states(game_to_postprocess, llm_outputs, "defender", turn)

    def _process_responses_and_game_states(self, game_to_postprocess, llm_outputs, player_role, turn):
        """Process model responses and update game states"""
        
        for (idx, game), output in zip(game_to_postprocess, llm_outputs):
            # Decode response
            response = self.tokenizer.batch_decode([output.outputs[0].token_ids], skip_special_tokens=True)[0]


            if not self.disable_hidden_cot:
                # Parse thinking and response
                (parsed_thinking, parsed_response), illgel_response_flag = cot_format_check_and_extract(response)
                
                # Compute length of parsed_response after tokenization
                if not illgel_response_flag:
                    thinking_text = parsed_thinking
                    thinking_encoded_len = len(self.tokenizer.encode(parsed_thinking, add_special_tokens=False))
                    answer_text = parsed_response
                    answer_encoded_len = len(self.tokenizer.encode(parsed_response, add_special_tokens=False))
                else:
                    thinking_text, answer_text = "", ""
                    thinking_encoded_len, answer_encoded_len = None, None
            else:
                parsed_response = response
            
            # Store output and metadata
            turn_states = {
                "game_role": player_role,
                "turn": turn,
                "game_idx": idx,
                "finished": game["finished"],
                "is_generated_attack": game.get("is_generated_attack", False),
                "prompt_type": game["prompt_type"],
                "prompts": game["prompts"],
                "completion": game["completion"],
            } # additional information for each turn to be used in making samples and later for experience

            if not self.disable_hidden_cot:
                turn_states.update({
                    "text_cot_and_answer": (thinking_text, answer_text),
                    "length_cot_and_answer": (thinking_encoded_len, answer_encoded_len),
                    "cot_format_violation": illgel_response_flag,
                })

            # Update game history
            game["history"].append({
                "game_role": player_role,
                "turn": turn,
                "content": parsed_response.strip(),
            })
            game["raw_history"].append({
                "game_role": player_role,
                "turn": turn,
                "content": response.strip()
            })
            game["processed_output_history"].append({
                "game_role": player_role,
                "turn": turn,
                "output": output,
                "game_states": turn_states,
            })
            
        
    def evaluate_game_outcomes(self):
        """Evaluate final game outcomes and get rewards"""
        assert self.remote_rm_url is not None, "Remote RM URL is not set"
        
        # Prepare batch queries for reward model
        batch_queries = []
        
        for idx, game in self.active_games.items():
            assert len(game["history"]) == 2, "Game should have 2 turns"
            attacker_move = game["history"][0]["content"]
            defender_move = game["history"][1]["content"]
            
            batch_queries.append({
                "game_idx": idx,
                "prompt": attacker_move,  # attacker move
                "response": defender_move  # defender move
            })
        
        # Get labels from remote reward model
        start_time = time.time()
        batch_labels_dict = remote_rm_fn(self.remote_rm_url[0], batch_queries, score_key="labels")
        end_time = time.time()
        self.strategy.print(f"Rank #{self.strategy.get_rank()}, Time taken to get labels: {round(end_time - start_time, 2)} secs")

        if isinstance(batch_labels_dict, list):
            batch_labels_dict = {idx: label for idx, label in enumerate(batch_labels_dict)}
        
        return batch_labels_dict
        
    def filter_and_compute_rewards(self, batch_labels_dict):
        """
        The role of this function is to filter out the games that are not valid and compute the rewards for the valid games.
        This is because sometimes the LLM generates gibberish responses that would fail the WildGuard classification.
        Additionally, since we could only compute the rewards after the game is finished, 
        we need to go back to the processed_output_history and add the rewards to the game states.
        """
        reward_coeff_config = self.custom_configs.get("reward_coeff_config", "general_sum")

        attacker_outputs = []
        attacker_turn_states = []
        defender_outputs = []
        defender_turn_states = []
        
        for game_idx, game in self.active_games.items():
            if game_idx not in batch_labels_dict:
                raise ValueError(f"Game {game_idx} not found in batch_labels_dict")
            
            labels = batch_labels_dict[game_idx]
            
            # Skip if no processed output history
            if not game["processed_output_history"]:
                continue

            # Skip if wildguard cannot parse the response
            if labels.get('is_parsing_error', False):
                continue
                
            for turn_idx, turn in enumerate(game["processed_output_history"]):
                output, turn_states = turn["output"], turn["game_states"]
                
                # Skip if no turn states
                if not turn_states:
                    assert game['is_generated_attack'] is False, "Generated attack should always have turn states"
                    continue

                # Compute rewards and update turn states
                reward, outcome = self.get_redteaming_game_reward(gamedata=turn_states, labels=labels, reward_coeff_config=reward_coeff_config)
                if not self.disable_hidden_cot:
                    reward += get_cot_formatting_reward(turn_states.get('cot_format_violation', None))
                
                # Update turn_states with computed values
                turn_states['reward'] = reward
                turn_states['game_outcomes'] = outcome
                
                # Sort outputs and states by role
                if turn_states["game_role"] == "attacker":
                    attacker_outputs.append(output)
                    attacker_turn_states.append(turn_states)
                elif turn_states["game_role"] == "defender":
                    defender_outputs.append(output)
                    defender_turn_states.append(turn_states)
                
        return attacker_outputs, attacker_turn_states, defender_outputs, defender_turn_states, batch_labels_dict