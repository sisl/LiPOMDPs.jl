
# Historical data access functions
function get_historical_price(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][2]  # price
    else
        @warn "No historical price data for year $year"
        return pomdp.p₀
    end
end

function get_historical_demand(pomdp::LiPOMDP, year::Int64)
    # Global production is the basis for company demand
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][1]  # production (used as demand basis)
    else
        @warn "No historical demand data for year $year"
        return 0.0
    end
end

function get_historical_production(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][1]  # production (t Li)
    elseif year > maximum(keys(pomdp.hist_data))
        # Only reached for the terminal state's next-year demand; hold the last observed year
        return pomdp.hist_data[maximum(keys(pomdp.hist_data))][1]
    else
        @warn "No historical production data for year $year"
        return 0.0
    end
end

# Helper function to get all historical data for a year
function get_historical_data(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        production, price = pomdp.hist_data[year]
        return (
            production=production,
            price=price
        )
    else
        @warn "No historical data for year $year"
        return (
            production=0.0,
            price=pomdp.p₀
        )
    end
end


function total_mined_so_far(P::LiPOMDP, initial_state::State, current_state::State)
    total = 0.0
    for site in 1:P.n_deposits
        mined_raw = initial_state.deposit[site] - current_state.deposit[site]
        total += mined_raw * P.recovery[site]  # nominal recovery
    end
    return total
end