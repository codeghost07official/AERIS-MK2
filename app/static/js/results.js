(() => {
  const dataElement = document.getElementById("analysis-chart-data");
  const telemetryCanvas = document.getElementById("telemetry-chart");
  const distributionCanvas = document.getElementById("distribution-chart");
  if (!dataElement || !telemetryCanvas || !distributionCanvas) return;

  const parsed = JSON.parse(dataElement.textContent || "{}");
  if (!window.Chart) {
    document.getElementById("telemetry-chart-fallback").hidden = false;
    document.getElementById("distribution-chart-fallback").hidden = false;
    return;
  }

  const palette = ["#ae8dff", "#76c9c0", "#f0bd78", "#f08da8",
    "#8fa9ff", "#cbb378", "#75aee8", "#c28be6"];
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: {
        labels: {
          color: "#c6bfd2",
          usePointStyle: true,
          boxWidth: 7,
          padding: 16,
        },
      },
      tooltip: { enabled: true },
    },
    scales: {
      x: {
        title: { display: true, text: "Record order", color: "#aaa3b8" },
        ticks: { color: "#8d869a", maxTicksLimit: 8 },
        grid: { color: "rgba(255,255,255,.055)" },
      },
      y: {
        title: { display: true, text: "Sensor value (unit not supplied)", color: "#aaa3b8" },
        ticks: { color: "#8d869a", maxTicksLimit: 6 },
        grid: { color: "rgba(255,255,255,.055)" },
      },
    },
  };

  const telemetryDatasets = parsed.series.map((series, index) => ({
    label: series.name,
    data: series.points,
    borderColor: palette[index % palette.length],
    backgroundColor: palette[index % palette.length],
    borderWidth: 1.7,
    pointRadius: 0,
    pointHitRadius: 7,
    tension: 0.12,
  }));
  if (parsed.prediction) {
    telemetryDatasets.push(
      {
        label: `${parsed.prediction.feature_name} observed (selected channel)`,
        data: parsed.prediction.actual,
        borderColor: "#f0bd78",
        backgroundColor: "#f0bd78",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0,
      },
      {
        label: `${parsed.prediction.feature_name} one-step prediction`,
        data: parsed.prediction.predicted,
        borderColor: "#8fa9ff",
        backgroundColor: "#8fa9ff",
        borderWidth: 1.5,
        pointRadius: 0,
        borderDash: [5, 4],
        tension: 0,
      },
      {
        type: "scatter",
        label: "Residual-threshold flag (not confirmed anomaly)",
        data: parsed.prediction.anomalies,
        borderColor: "#f08da8",
        backgroundColor: "#f08da8",
        pointRadius: 4,
      },
    );
  }

  new window.Chart(telemetryCanvas, {
    type: "line",
    data: { datasets: telemetryDatasets },
    options: {
      ...chartDefaults,
      parsing: false,
      scales: {
        ...chartDefaults.scales,
        x: { ...chartDefaults.scales.x, type: "linear" },
      },
    },
  });

  const select = document.getElementById("sensor-select");
  const selectedSensor = () => parsed.sensors[Number(select.value)];
  const histogramData = (sensor) => ({
    labels: sensor.histogram.labels,
    counts: sensor.histogram.counts,
  });
  const sensor = selectedSensor();
  const distributionChart = new window.Chart(distributionCanvas, {
    type: "bar",
    data: {
      labels: histogramData(sensor).labels,
      datasets: [{
        label: "Records",
        data: histogramData(sensor).counts,
        backgroundColor: "rgba(169,138,255,.68)",
        borderColor: "#ae8dff",
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      ...chartDefaults,
      plugins: {
        ...chartDefaults.plugins,
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Value range: ${items[0].label}`,
            label: (item) => `${item.raw} record(s)`,
          },
        },
      },
      scales: {
        ...chartDefaults.scales,
        x: {
          ...chartDefaults.scales.x,
          title: { display: true, text: "Sensor value range", color: "#aaa3b8" },
          ticks: { color: "#8d869a", maxRotation: 45, minRotation: 0, maxTicksLimit: 6 },
        },
        y: {
          ...chartDefaults.scales.y,
          title: { display: true, text: "Records", color: "#aaa3b8" },
          beginAtZero: true,
        },
      },
    },
  });

  select.addEventListener("change", () => {
    const selected = selectedSensor();
    distributionChart.data.labels = selected.histogram.labels;
    distributionChart.data.datasets[0].data = selected.histogram.counts;
    distributionChart.data.datasets[0].label = `${selected.name} records`;
    distributionChart.update();
  });
})();
