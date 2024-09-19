let svg = d3.select('body').append('svg')
    .attr('width', window.innerWidth)
    .attr('height', window.innerHeight);

let stateData = new Map();
let path = d3.geoPath();

let mapUrl = 'path/to/your/topojson.json';  // Your TopoJSON map URL
let dataUrl = 'data/poll_data_winner.csv';  // Your CSV file path

let div = d3.select('body').append('div')
    .attr('class', 'tooltip')
    .style('opacity', 0);

d3.queue()
    .defer(d3.json, mapUrl)  // Load TopoJSON
    .defer(d3.csv, dataUrl, function(d) {
        // Parse the CSV and store in map by state name
        stateData.set(d.state, {
            winner: d.candidate_name,
            pct: +d.weighted_avg_pct
        });
    })
    .await(function(error, topoData) {
        if (error) throw error;

        // Convert TopoJSON to GeoJSON
        const states = topojson.feature(topoData, topoData.objects.states).features;

        // Draw the states
        svg.append('g')
            .selectAll('path')
            .data(states)
            .enter().append('path')
            .attr('d', path)  // Create the geographic paths for the states
            .attr('class', 'state')  // Set class for each state path
            .attr('fill', function(d) {
                let stateName = d.properties.name;  // Use the 'name' field in properties
                let data = stateData.get(stateName);

                // Set fill color based on the winner
                if (!data) return 'gray';  // No data, fill gray

                // Color coding based on party (you can adjust colors)
                if (data.winner === 'Donald Trump') return '#f44336';  // Red for GOP
                else if (data.winner === 'Joe Biden' || data.winner === 'Kamala Harris') return '#2196f3';  // Blue for Dems
                else return 'gray';  // Gray for other parties
            })
            .on('mouseover', function(event, d) {
                let stateName = d.properties.name;
                let data = stateData.get(stateName);

                div.transition()
                    .duration(200)
                    .style('opacity', .9);

                // Tooltip content: State name and projected winner
                div.html(`<strong>${stateName}</strong><br>Winner: ${data ? data.winner : 'N/A'}`)
                    .style('left', (event.pageX + 5) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function() {
                div.transition().duration(500).style('opacity', 0);
            });

        // Draw state borders
        svg.append('g')
            .selectAll('path')
            .data(states)
            .enter().append('path')
            .attr('d', path)
            .attr('fill', 'none')
            .attr('stroke', '#000')  // Black borders
            .attr('stroke-width', '1px');
    });