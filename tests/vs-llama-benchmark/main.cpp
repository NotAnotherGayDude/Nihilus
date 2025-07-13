#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <../src/spinlock_time.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include <../src/llama-context.h>
#include <nihilus/index.hpp>
#include <llama.h>
#include "common/arg.h"
#include "common/chat.h"
#include "common/common.h"
#include "common/console.h"
#include "common/log.h"
#include "common/sampling.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
	#include <signal.h>
	#include <unistd.h>
#elif defined(_WIN32)
	#include <signal.h>
	#include <windows.h>
#endif

#if defined(_MSC_VER)
	#pragma warning(disable : 4244 4267)// possible loss of data
#endif

static llama_context** g_ctx;
static llama_model** g_model;
static common_sampler** g_smpl;
static common_params* g_params;
static std::vector<llama_token>* g_input_tokens;
static std::ostringstream* g_output_ss;
static std::vector<llama_token>* g_output_tokens;
static bool is_interacting	= false;
static bool need_insert_eot = false;

static void print_usage(int argc, char** argv) {
	( void )argc;

	LOG("\nexample usage:\n");
	LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
	LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
	LOG("\n");
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
static void sigint_handler(int signo) {
	if (signo == SIGINT) {
		if (!is_interacting && g_params->interactive) {
			is_interacting	= true;
			need_insert_eot = true;
		} else {
			console::cleanup();
			LOG("\n");
			common_perf_print(*g_ctx, *g_smpl);

			// make sure all logs are flushed
			LOG("Interrupted by user\n");
			common_log_pause(common_log_main());

			_exit(130);
		}
	}
}
#endif

#if !defined(LLAMA_MODEL_SIZE)
static constexpr nihilus::model_sizes model_size{ nihilus::model_sizes::llm_8B };
#else
static constexpr nihilus::model_sizes model_size{ LLAMA_MODEL_SIZE };
#endif

int main(int argc, char** argv) {
	try {
		static constexpr auto model_config = nihilus::generate_model_config(nihilus::model_generations::v3, model_size, nihilus::kernel_type_profiles::q8_gqa,
			nihilus::model_arches::llama, false);
		static constexpr auto model_config01 = nihilus::update_model_config_benchmark(model_config, true);
		nihilus::cli_params cli_args_final	 = nihilus::harbinger<model_config01>::parse_cli_arguments(argc, argv);
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::runBenchmark<"nihilus">([&] {
			auto model_new{ nihilus::harbinger<model_config01>::parse_model_graph_data(cli_args_final) };
			while (model_new->process_input(cli_args_final.prompt)) {
			}
			bnch_swt::doNotOptimizeAway(cli_args_final.n_tokens);
			return cli_args_final.n_tokens;
		});
		test::stop_watch stop_watch_val{ 0 };
		std::string return_value{};
		common_params params;
		llama_context* ctx{};
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::runBenchmark<"llama.cpp">([&] {
			return_value.clear();
			uint64_t token_count{ cli_args_final.n_tokens };
			g_params = &params;
			if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
				return 1;
			}

			common_init();

			auto& sparams = params.sampling;

			// save choice to use color for later
			// (note for later: this is a slightly awkward choice)
			console::init(params.simple_io, params.use_color);

			LOG_INF("%s: llama backend init\n", __func__);

			llama_backend_init();
			llama_numa_init(params.numa);

			llama_model* model	 = nullptr;
			ctx	 = nullptr;
			common_sampler* smpl = nullptr;

			g_model = &model;
			g_ctx	= &ctx;
			g_smpl	= &smpl;
			std::vector<common_chat_msg> chat_msgs;
			// load the model and apply lora adapter, if any
			LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
			common_init_result llama_init = common_init_from_params(params);

			model = llama_init.model.get();
			ctx	  = llama_init.context.get();
			if (model == NULL) {
				LOG_ERR("%s: error: unable to load model\n", __func__);
				return 1;
			}

			const llama_vocab* vocab = llama_model_get_vocab(model);
			auto chat_templates		 = common_chat_templates_init(model, params.chat_template);

			LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, ( int )params.cpuparams.n_threads);

			auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
			if (!cpu_dev) {
				LOG_ERR("%s: no CPU backend found\n", __func__);
				return 1;
			}
			auto* reg					  = ggml_backend_dev_backend_reg(cpu_dev);
			auto* ggml_threadpool_new_fn  = ( decltype(ggml_threadpool_new)* )ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
			auto* ggml_threadpool_free_fn = ( decltype(ggml_threadpool_free)* )ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

			struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
			struct ggml_threadpool_params tpp		= ggml_threadpool_params_from_cpu_params(params.cpuparams);

			set_process_priority(params.cpuparams.priority);

			struct ggml_threadpool* threadpool_batch = NULL;
			if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
				threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
				if (!threadpool_batch) {
					LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
					return 1;
				}

				// Start the non-batch threadpool in the paused state
				tpp.paused = true;
			}

			struct ggml_threadpool* threadpool = ggml_threadpool_new_fn(&tpp);
			if (!threadpool) {
				LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
				return 1;
			}

			llama_attach_threadpool(ctx, threadpool, threadpool_batch);
			const uint32_t n_ctx = llama_n_ctx(ctx);

			// print system information
			{
				LOG_INF("\n");
				LOG_INF("%s\n", common_params_get_system_info(params).c_str());
				LOG_INF("\n");
			}

			std::string path_session = params.path_prompt_cache;
			std::vector<llama_token> session_tokens;

			ctx->stop_watch_val = stop_watch_val;
			const bool add_bos	= llama_vocab_get_add_bos(vocab) && !params.use_jinja;
			if (!llama_model_has_encoder(model)) {
				GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
			}

			LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

			std::vector<llama_token> embd_inp;

			bool waiting_for_first_input = false;
			auto chat_add_and_format	 = [&chat_msgs, &chat_templates](const std::string& role, const std::string& content) {
				common_chat_msg new_msg;
				new_msg.role	= role;
				new_msg.content = content;
				auto formatted	= common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
				chat_msgs.push_back(new_msg);
				LOG_DBG("formatted: '%s'\n", formatted.c_str());
				return formatted;
			};

			std::string prompt;
			{
				prompt = params.prompt;

				LOG_DBG("tokenize the prompt\n");
				embd_inp = common_tokenize(ctx, prompt, true, true);

				LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
				LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
			}

			// Tokenize negative prompt
			if (( int )embd_inp.size() > n_ctx - 4) {
				LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, ( int )embd_inp.size(), n_ctx - 4);
				return 1;
			}

			// debug message about similarity of saved session, if applicable
			uint64_t n_matching_session_tokens = 0;

			// ctrl+C handling
			{
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
				struct sigaction sigint_action;
				sigint_action.sa_handler = sigint_handler;
				sigemptyset(&sigint_action.sa_mask);
				sigint_action.sa_flags = 0;
				sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
				auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
					return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
				};
				SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
			}

			smpl = common_sampler_init(model, sparams);
			if (!smpl) {
				LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
				return 1;
			}

			LOG_INF("sampler seed: %u\n", common_sampler_get_seed(smpl));
			LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
			LOG_INF("sampler chain: %s\n", common_sampler_print(smpl).c_str());

			LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

			const int ga_n = params.grp_attn_n;
			const int ga_w = params.grp_attn_w;
			LOG_INF("\n");

			bool is_antiprompt		  = false;
			bool input_echo			  = true;
			bool display			  = true;
			bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

			int n_past	   = 0;
			int n_remain   = params.n_predict;
			int n_consumed = 0;

			std::vector<int> input_tokens;
			g_input_tokens = &input_tokens;
			std::vector<int> output_tokens;
			g_output_tokens = &output_tokens;
			std::ostringstream output_ss;
			g_output_ss = &output_ss;
			std::ostringstream assistant_ss;// for storing current assistant message, used in conversation mode

			// the first thing we will do is to output the prompt, so set color accordingly
			console::set_display(console::prompt);
			display = params.display_prompt;

			std::vector<llama_token> embd;

			// single-token antiprompts
			std::vector<llama_token> antiprompt_token;

			while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
				++token_count;
				// predict
				if (!embd.empty()) {
					// Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
					// --prompt or --file which uses the same value.
					uint32_t max_embd_size = n_ctx - 4;

					for (int i = 0; i < ( int )embd.size(); i += params.n_batch) {
						int n_eval = ( int )embd.size() - i;
						if (n_eval > params.n_batch) {
							n_eval = params.n_batch;
						}

						LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());
						//stop_watch_val.reset();
						if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
							LOG_ERR("%s : failed to eval\n", __func__);
							return 1;
						}
						//stop_watch_val.add_time();

						n_past += n_eval;

						LOG_DBG("n_past = %d\n", n_past);
					}
				}

				embd.clear();

				if (( int )embd_inp.size() <= n_consumed && !is_interacting) {
					// optionally save the session on first sample (for faster prompt loading next time)
					if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
						need_to_save_session = false;
						llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

						LOG_DBG("saved session to %s\n", path_session.c_str());
					}

					const llama_token id = common_sampler_sample(smpl, ctx, -1);

					common_sampler_accept(smpl, id, true);

					// LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

					embd.push_back(id);

					// echo this to console
					input_echo = true;

					// decrement remaining sampling budget
					--n_remain;

					LOG_DBG("n_remain: %d\n", n_remain);
				} else {
					// some user input remains from prompt or interaction, forward it to processing
					LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", ( int )embd_inp.size(), n_consumed);
					while (( int )embd_inp.size() > n_consumed) {
						embd.push_back(embd_inp[n_consumed]);

						// push the prompt in the sampling context in order to apply repetition penalties later
						// for the prompt, we don't apply grammar rules
						common_sampler_accept(smpl, embd_inp[n_consumed], false);

						++n_consumed;
						if (( int )embd.size() >= params.n_batch) {
							break;
						}
					}
				}

				// display text
				if (input_echo && display) {
					for (auto id: embd) {
						const std::string token_str = common_token_to_piece(ctx, id, params.special);

						// Console/Stream Output
						return_value.append(token_str.c_str());
						//LOG("%s", token_str.c_str());

						// Record Displayed Tokens To Log
						// Note: Generated tokens are created one by one hence this check
						if (embd.size() > 1) {
							// Incoming Requested Tokens
							input_tokens.push_back(id);
						} else {
							// Outgoing Generated Tokens
							output_tokens.push_back(id);
							output_ss << token_str;
						}
					}
				}

				// reset color to default if there is no pending user input
				if (input_echo && ( int )embd_inp.size() == n_consumed) {
					console::set_display(console::reset);
					display = true;
				}

				// if not currently processing queued inputs;
				if (( int )embd_inp.size() <= n_consumed) {
					// check for reverse prompt in the last n_prev tokens
					if (!params.antiprompt.empty()) {
						const int n_prev			  = 32;
						const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

						is_antiprompt = false;
						// Check if each of the reverse prompts appears at the end of the output.
						// If we're not running interactively, the reverse prompt might be tokenized with some following characters
						// so we'll compensate for that by widening the search window a bit.
						for (std::string& antiprompt: params.antiprompt) {
							uint64_t extra_padding	  = params.interactive ? 0 : 2;
							uint64_t search_start_pos = last_output.length() > static_cast<uint64_t>(antiprompt.length() + extra_padding)
								? last_output.length() - static_cast<uint64_t>(antiprompt.length() + extra_padding)
								: 0;

							if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
								if (params.interactive) {
									is_interacting = true;
								}
								is_antiprompt = true;
								break;
							}
						}

						// check for reverse prompt using special tokens
						llama_token last_token = common_sampler_last(smpl);
						for (auto token: antiprompt_token) {
							if (token == last_token) {
								if (params.interactive) {
									is_interacting = true;
								}
								is_antiprompt = true;
								break;
							}
						}

						if (is_antiprompt) {
							LOG_DBG("found antiprompt: %s\n", last_output.c_str());
						}
					}

					// deal with end of generation tokens in interactive mode
					if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
						LOG_DBG("found an EOG token\n");

						if (params.interactive) {
							if (!params.antiprompt.empty()) {
								// tokenize and inject first reverse prompt
								const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
								embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
								is_antiprompt = true;
							}

							if (params.enable_chat_template) {
								chat_add_and_format("assistant", assistant_ss.str());
							}
							is_interacting = true;
							LOG("\n");
						}
					}

					// if current token is not EOG, we add it to current assistant message
					if (params.conversation_mode && !waiting_for_first_input) {
						const auto id = common_sampler_last(smpl);
						assistant_ss << common_token_to_piece(ctx, id, false);

						if (!prompt.empty()) {
							prompt.clear();
							is_interacting = false;
						}
					}

					if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
						LOG_DBG("waiting for user input\n");

						if (params.conversation_mode) {
							LOG("\n> ");
						}

						if (params.input_prefix_bos) {
							LOG_DBG("adding input prefix BOS token\n");
							embd_inp.push_back(llama_vocab_bos(vocab));
						}

						std::string buffer;
						if (!params.input_prefix.empty() && !params.conversation_mode) {
							LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
							LOG("%s", params.input_prefix.c_str());
						}

						// color user input only
						console::set_display(console::user_input);
						display = params.display_prompt;

						std::string line;
						bool another_line = true;
						do {
							another_line = console::readline(line, params.multiline_input);
							buffer += line;
						} while (another_line);

						// done taking input, reset color
						console::set_display(console::reset);
						display = true;

						if (buffer.empty()) {// Ctrl+D on empty line exits
							LOG("EOF by user\n");
							break;
						}

						if (buffer.back() == '\n') {
							// Implement #587:
							// If the user wants the text to end in a newline,
							// this should be accomplished by explicitly adding a newline by using \ followed by return,
							// then returning control by pressing return again.
							buffer.pop_back();
						}

						if (buffer.empty()) {// Enter key on empty line lets the user pass control back
							LOG_DBG("empty line, passing control back\n");
						} else {// Add tokens to embd only if the input buffer is non-empty
							// append input suffix if any
							if (!params.input_suffix.empty() && !params.conversation_mode) {
								LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
								LOG("%s", params.input_suffix.c_str());
							}

							LOG_DBG("buffer: '%s'\n", buffer.c_str());

							const uint64_t original_size = embd_inp.size();

							if (params.escape) {
								string_process_escapes(buffer);
							}

							bool format_chat	 = params.conversation_mode && params.enable_chat_template;
							std::string user_inp = format_chat ? chat_add_and_format("user", move(buffer)) : move(buffer);
							// TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
							const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
							const auto line_inp = common_tokenize(ctx, user_inp, false, format_chat);
							const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

							LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

							// if user stop generation mid-way, we must add EOT to finish model's last response
							if (need_insert_eot && format_chat) {
								llama_token eot = llama_vocab_eot(vocab);
								embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
								need_insert_eot = false;
							}

							embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
							embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
							embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

							for (uint64_t i = original_size; i < embd_inp.size(); ++i) {
								const llama_token token = embd_inp[i];
								output_tokens.push_back(token);
								output_ss << common_token_to_piece(ctx, token);
							}

							// reset assistant message
							assistant_ss.str("");

							n_remain -= line_inp.size();
							LOG_DBG("n_remain: %d\n", n_remain);
						}

						input_echo = false;// do not echo this again
					}

					if (n_past > 0 || waiting_for_first_input) {
						if (is_interacting) {
							common_sampler_reset(smpl);
						}
						is_interacting = false;

						if (waiting_for_first_input && params.single_turn) {
							params.interactive		 = false;
							params.interactive_first = false;
						}
						waiting_for_first_input = false;
					}
				}

				// end of generation
				if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
					LOG(" [end of text]\n");
					break;
				}

				// In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
				// We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
				if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
					n_remain	   = params.n_predict;
					is_interacting = true;
				}
			}

			LOG("\n\n");
			common_perf_print(ctx, smpl);

			common_sampler_free(smpl);

			stop_watch_val = ctx->stop_watch_val;
			llama_backend_free();
			ggml_threadpool_free_fn(threadpool);
			ggml_threadpool_free_fn(threadpool_batch);
			bnch_swt::doNotOptimizeAway(cli_args_final.n_tokens);
			return static_cast<int32_t>(cli_args_final.n_tokens);
		});

		std::cout << return_value << std::endl;

		std::cout << "FOR " << cli_args_final.thread_count << " THREADS, WITH " << spinlock_time << " NANOSECONDS OF SPINLOCK PER KERNEL, "
				  << "LLAMA.CPP/GGML AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("LLAMA.CPP/GGML AVERAGE COMPUTE TIME, OVER: ")) << stop_watch_val.get_count()
				  << " TOKENS: " << stop_watch_val.get_average() << std::endl;
		std::cout << "FOR " << cli_args_final.thread_count << " THREADS, WITH " << nihilus::spinlock_time << " NANOSECONDS OF SPINLOCK PER KERNEL, "
				  << "NIHILUS AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("NIHILUS AVERAGE COMPUTE TIME, OVER: ")) << nihilus::stop_watch_val_nihilus.get_count()
				  << " TOKENS: " << nihilus::stop_watch_val_nihilus.get_average() << std::endl;
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::printResults();
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}