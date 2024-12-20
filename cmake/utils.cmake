macro(append_to_cached_variable VAR_NAME)
    # Append the new arguments to the existing value
    set(new_value
            ${${VAR_NAME}}
            ${ARGN}
    )

    # Update the cache with the new value
    set(${VAR_NAME} ${new_value} CACHE INTERNAL "")
endmacro()
