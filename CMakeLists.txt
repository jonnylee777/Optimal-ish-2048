cmake_minimum_required(VERSION 3.20)

project(adversarial_2048 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADVERSARIAL_2048_BUILD_BENCHMARKS "Build engine benchmarks" ON)

function(adversarial_2048_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    endif()
endfunction()

add_library(adversarial_2048_core
    src/core/board.cpp
    src/core/move_tables.cpp
    src/core/spawn.cpp
)

target_include_directories(adversarial_2048_core PUBLIC src)
adversarial_2048_enable_warnings(adversarial_2048_core)

add_library(adversarial_2048_runtime
    src/agents/greedy_agent.cpp
    src/agents/random_agent.cpp
    src/agents/search_agent.cpp
    src/evaluation/baseline_heuristic.cpp
    src/evaluation/h0_heuristic.cpp
    src/evaluation/h2_heuristic.cpp
    src/evaluation/h3_heuristic.cpp
    src/evaluation/h4_heuristic.cpp
    src/evaluation/h5_heuristic.cpp
    src/evaluation/structural_heuristic.cpp
    src/experiments/game_runner.cpp
    src/experiments/result_writer.cpp
    src/experiments/run_experiment_config.cpp
    src/game/game.cpp
    src/learning/ntuple_network.cpp
    src/learning/td_trainer.cpp
    src/learning/temporal_coherence.cpp
    src/learning/position_store.cpp
    src/learning/neural_network.cpp
    src/search/parallel_expectimax.cpp
    src/evaluation/ensemble_evaluator.cpp
    src/optimization/baseline_optimizer.cpp
    src/optimization/structural_optimizer.cpp
    src/search/expectimax.cpp
    src/tablebase/formation.cpp
    src/tablebase/layer_store.cpp
    src/tablebase/solver.cpp
    src/tablebase/variant_mover.cpp
)
target_link_libraries(adversarial_2048_runtime PUBLIC adversarial_2048_core)
adversarial_2048_enable_warnings(adversarial_2048_runtime)

add_executable(adversarial_2048 src/main.cpp)
target_link_libraries(adversarial_2048 PRIVATE adversarial_2048_runtime)
target_compile_definitions(adversarial_2048 PRIVATE
    "ADVERSARIAL_2048_BUILD_TYPE=\"$<CONFIG>\""
)
adversarial_2048_enable_warnings(adversarial_2048)

add_executable(run_experiment src/run_experiment_main.cpp)
target_link_libraries(run_experiment PRIVATE adversarial_2048_runtime)
target_compile_definitions(run_experiment PRIVATE
    "ADVERSARIAL_2048_BUILD_TYPE=\"$<CONFIG>\""
)
adversarial_2048_enable_warnings(run_experiment)

add_executable(train_ntuple src/train_ntuple_main.cpp)
target_link_libraries(train_ntuple PRIVATE adversarial_2048_runtime)
target_compile_definitions(train_ntuple PRIVATE
    "ADVERSARIAL_2048_BUILD_TYPE=\"$<CONFIG>\""
)
adversarial_2048_enable_warnings(train_ntuple)

add_executable(optimize_baseline src/optimize_main.cpp)
target_link_libraries(optimize_baseline PRIVATE adversarial_2048_runtime)
target_compile_definitions(optimize_baseline PRIVATE
    "ADVERSARIAL_2048_BUILD_TYPE=\"$<CONFIG>\""
)
adversarial_2048_enable_warnings(optimize_baseline)

add_executable(optimize_structural src/optimize_structural_main.cpp)
target_link_libraries(optimize_structural PRIVATE adversarial_2048_runtime)
target_compile_definitions(optimize_structural PRIVATE
    "ADVERSARIAL_2048_BUILD_TYPE=\"$<CONFIG>\""
)
adversarial_2048_enable_warnings(optimize_structural)

include(CTest)

if(BUILD_TESTING)
    add_executable(engine_tests tests/engine_tests.cpp)
    target_link_libraries(engine_tests PRIVATE adversarial_2048_core)
    adversarial_2048_enable_warnings(engine_tests)
    add_test(NAME engine_tests COMMAND engine_tests)

    add_executable(milestone2_tests tests/milestone2_tests.cpp)
    target_link_libraries(milestone2_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(milestone2_tests)
    add_test(NAME milestone2_tests COMMAND milestone2_tests)

    add_executable(milestone3_tests tests/milestone3_tests.cpp)
    target_link_libraries(milestone3_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(milestone3_tests)
    add_test(NAME milestone3_tests COMMAND milestone3_tests)

    add_executable(milestone4_tests tests/milestone4_tests.cpp)
    target_link_libraries(milestone4_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(milestone4_tests)
    add_test(NAME milestone4_tests COMMAND milestone4_tests)

    add_executable(milestone5_tests tests/milestone5_tests.cpp)
    target_link_libraries(milestone5_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(milestone5_tests)
    add_test(NAME milestone5_tests COMMAND milestone5_tests)

    add_executable(milestone6_tests tests/milestone6_tests.cpp)
    target_link_libraries(milestone6_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(milestone6_tests)
    add_test(NAME milestone6_tests COMMAND milestone6_tests)

    add_executable(h0_heuristic_tests tests/h0_heuristic_tests.cpp)
    target_link_libraries(h0_heuristic_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(h0_heuristic_tests)
    add_test(NAME h0_heuristic_tests COMMAND h0_heuristic_tests)

    add_executable(h2_heuristic_tests tests/h2_heuristic_tests.cpp)
    target_link_libraries(h2_heuristic_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(h2_heuristic_tests)
    add_test(NAME h2_heuristic_tests COMMAND h2_heuristic_tests)

    add_executable(h3_heuristic_tests tests/h3_heuristic_tests.cpp)
    target_link_libraries(h3_heuristic_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(h3_heuristic_tests)
    add_test(NAME h3_heuristic_tests COMMAND h3_heuristic_tests)

    add_executable(h4_heuristic_tests tests/h4_heuristic_tests.cpp)
    target_link_libraries(h4_heuristic_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(h4_heuristic_tests)
    add_test(NAME h4_heuristic_tests COMMAND h4_heuristic_tests)

    add_executable(h5_heuristic_tests tests/h5_heuristic_tests.cpp)
    target_link_libraries(h5_heuristic_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(h5_heuristic_tests)
    add_test(NAME h5_heuristic_tests COMMAND h5_heuristic_tests)

    add_executable(tablebase_formation_tests tests/tablebase_formation_tests.cpp)
    target_link_libraries(tablebase_formation_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(tablebase_formation_tests)
    add_test(NAME tablebase_formation_tests COMMAND tablebase_formation_tests)

    add_executable(tablebase_solver_tests tests/tablebase_solver_tests.cpp)
    target_link_libraries(tablebase_solver_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(tablebase_solver_tests)
    add_test(NAME tablebase_solver_tests COMMAND tablebase_solver_tests)

    add_executable(tablebase_disk_tests tests/tablebase_disk_tests.cpp)
    target_link_libraries(tablebase_disk_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(tablebase_disk_tests)
    add_test(NAME tablebase_disk_tests COMMAND tablebase_disk_tests)

    add_executable(ntuple_network_tests tests/ntuple_network_tests.cpp)
    target_link_libraries(ntuple_network_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(ntuple_network_tests)
    add_test(NAME ntuple_network_tests COMMAND ntuple_network_tests)

    add_executable(td_trainer_tests tests/td_trainer_tests.cpp)
    target_link_libraries(td_trainer_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(td_trainer_tests)
    add_test(NAME td_trainer_tests COMMAND td_trainer_tests)

    add_executable(evaluation_semantics_tests tests/evaluation_semantics_tests.cpp)
    target_link_libraries(evaluation_semantics_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(evaluation_semantics_tests)
    add_test(NAME evaluation_semantics_tests COMMAND evaluation_semantics_tests)

    add_executable(temporal_coherence_tests tests/temporal_coherence_tests.cpp)
    target_link_libraries(temporal_coherence_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(temporal_coherence_tests)
    add_test(NAME temporal_coherence_tests COMMAND temporal_coherence_tests)

    add_executable(run_experiment_tests tests/run_experiment_tests.cpp)
    target_link_libraries(run_experiment_tests PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(run_experiment_tests)
    add_test(NAME run_experiment_tests COMMAND run_experiment_tests)
endif()

if(ADVERSARIAL_2048_BUILD_BENCHMARKS)
    add_executable(engine_benchmark benchmarks/engine_benchmark.cpp)
    target_link_libraries(engine_benchmark PRIVATE adversarial_2048_core)
    adversarial_2048_enable_warnings(engine_benchmark)

    add_executable(runtime_benchmark benchmarks/runtime_benchmark.cpp)
    target_link_libraries(runtime_benchmark PRIVATE adversarial_2048_runtime)
    adversarial_2048_enable_warnings(runtime_benchmark)
endif()
