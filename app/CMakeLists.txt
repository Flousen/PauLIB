add_executable(main
    main.cpp
)

target_link_libraries(main
    PRIVATE
        paulib
)

target_compile_options(main PRIVATE ${PAULIB_WARNINGS} ${PAULIB_VECTORIZE_FLAGS})
