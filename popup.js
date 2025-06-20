document.getElementById('check-btn').addEventListener('click', async () => {
  const text = document.getElementById('input-text').value.trim();
  const resultDiv = document.getElementById('result');
  const loading = document.getElementById('loading'); // 关键修复点

    // 隐藏结果，显示加载动画
  resultDiv.style.display = 'none';
  loading.style.display = 'flex'

  if (!text) {
    loading.style.display = 'none';
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<p style="color:red">请输入要检测的新闻内容</p>';
    return;
  }
  
    // 检测文本中的词语数量（按空格分割）
  
  if (text.length < 20) {
      loading.style.display = 'none';
      resultDiv.style.display = 'block';
      resultDiv.innerHTML = '<p style="color:red">请输入新闻完整内容进行分析</p>';
      return;
  }
  try {
    // 发送 POST 请求
    const response = await fetch('http://localhost:5000/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });
    // 检查响应状态
    if (!response.ok) throw new Error('请求失败');
    // 解析响应数据
    const data = await response.json();
    // 处理响应数据
    resultDiv.className = getResultClass(data);
    
      

    
    // 更新基础信息

    document.getElementById('Media_sources').textContent =
      data.Media_sources;
    document.getElementById('ai_result').textContent =
      data.is_AI ? ' 疑似生成式人工智能撰写' : ' 可能为人工撰写';

    const objectivityScoreElement = document.getElementById('objectivity-score');
    const commentElement = document.getElementById('objectivity-comment');
    if (objectivityScoreElement) {
      objectivityScoreElement.textContent = data.objectivity_score;
      // 根据得分设置样式
      const score = parseFloat(parseFloat(data.objectivity_score).toFixed(2));
      if (score >= 0 && score <= 45) {
        objectivityScoreElement.style.color ='red';
        commentElement.textContent = '文本客观性有待研究，请谨慎阅读';
      } else if (score >45 && score <= 75) {
        objectivityScoreElement.style.color = 'darkyellow';
        commentElement.textContent = '文本内容客观性一般，请理性阅读';
      } else if (score > 75 && score <= 100) {
        objectivityScoreElement.style.color = 'green';
        commentElement.textContent = '内容客观性较强';
      }
      // 添加鼠标悬停和移开事件
      objectivityScoreElement.addEventListener('mouseenter', () => {
        commentElement.style.opacity = 1;
        commentElement.style.visibility = 'visible';
      });

      objectivityScoreElement.addEventListener('mouseleave', () => {
        commentElement.style.opacity = 0;
        commentElement.style.visibility = 'hidden';
      });
    }
     // 绘制饼图
    drawPieChart(data.probability, data.is_fake);  // 修改点：传入 is_fake
    // 显示结果，隐藏加载动画
    loading.style.display = 'none';
    resultDiv.style.display = 'block';
    document.querySelector('.charts--container').style.display = 'flex'; // 显示容器


    // 显示词云
    const wordcloudImg = document.getElementById('wordcloud-img');
    wordcloudImg.src = data.wordcloud;
    wordcloudImg.style.display = 'block';

  } catch (error) {
    loading.style.display = 'none';
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `<p style="color:red">检测失败: ${error.message}</p>`;
  }
});

// 饼图绘制函数
function drawPieChart(probability, is_fake) {
  const width = 300;
  const height = 200;
  const radius = Math.min(width, height) / 2;
  const container = d3.select("#pieChartSVG")
    .attr("width", width)
    .attr("height", height);

  // 清空旧图表
  container.selectAll("*").remove();
   // 根据 is_fake 动态分配颜色（核心修改）
  const fakeColor = '#ff4444' ;  // 时虚假部分为红，
  const  realColor = '#2196F3';   // 真实部分为蓝


  // 创建数据
  const pieData = [
    { 
      color: fakeColor  , 
      description: '虚假新闻', 
      value: is_fake ? probability / 100 : (100 - probability) / 100  
    },
    { 
      color: realColor, 
      description: '真实新闻', 
      value: is_fake ? (100 - probability) / 100 : probability / 100
    }
  ];

  // 创建饼图布局
  const pie = d3.pie().value(d => d.value).sort(null);

   // 修改为圆环效果（核心修改）
  const arc = d3.arc()
    .outerRadius(radius - 10)
    .innerRadius(radius * 0.6);  // 调整内半径形成圆环

  // 创建组元素
  const arcs = container.append("g")
    .attr("transform", `translate(${width/2},${height/2})`)
    .selectAll("path")
    .data(pie(pieData))
    .enter()
    .append("g");

  // 绘制扇形
  arcs.append("path")
    .attr("fill", d => d.data.color)
    .transition()
    .duration(1000)
    .attrTween("d", d => {
      const interpolate = d3.interpolate({startAngle: 0, endAngle: 0}, d);
      return t => arc(interpolate(t));
    });

  // 添加中心文字
  const centerGroup = container.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  centerGroup.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", "-0.2em")  // 调整垂直位置
    .style("font-size", "24px")
    .style("fill", "black")//设置piechart中文字的颜色
    .style("font-weight", "bold")
    .text("检测结果");

  centerGroup.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", "1.0em")
    .style("font-size", "32px")
    .style("fill", "black")//设置piechart中文字的颜色
    .text(`${probability}%`);

  // 添加图例
  const legend = container.append("g")
     .attr("transform", `translate(${width/2 - 100},${height + 10})`); // 居中下方


  
  // 调整图例项顺序
  const legendData = pieData.reverse();  // 反转顺序

  legend.selectAll("rect")
    .data(legendData)
    .enter()
    .append("rect")
    .attr("x", (d,i) => i * 150)  // 增加间距
    .attr("width", 18)
    .attr("height", 18)
    .style("fill", d => d.color);

  legend.selectAll("text")
    .data(legendData)
    .enter()
    .append("text")
    .attr("x", (d,i) => i * 150 +20 )
    .attr("y", 14)
    .style("fill", "black")//设置piechart中文字的颜色
    .text(d => `${d.description} (${(d.value*100).toFixed(1)}%)`);
}


// 根据多个条件设置结果区域的样式类
function getResultClass(data) {
  const { is_fake, probability, is_AI, Media_sources, objectivity_score } = data;
  const probThreshold = 65; // 概率阈值
  
  // 辅助判断函数
  const hasNoMediaSources =  Media_sources.trim() === "无来源";
  const isObjectivityLow = parseFloat(objectivity_score) < 60;
  
  // 当 is_fake 为 1 时
  if (is_fake === 1) {
    if (probability < probThreshold) {
      // 条件1: 任一条件满足则为 fake
      if (is_AI === 1 || hasNoMediaSources || isObjectivityLow) {
        document.getElementById('My_advice').textContent ='请谨慎阅读文本内容，切勿轻易相信';
        return 'fake';
      }
      // 条件2: 所有条件不满足则为 consider
      else {
        document.getElementById('My_advice').textContent ='阅读文本内容时保持主观思考';
        return 'consider';
      }
    }
    // 条件3: probability >= 0.65 直接为 fake
    else {
      document.getElementById('My_advice').textContent ='此文本内容虚假';
      return 'fake';
    }
  }
  // 当 is_fake 为 0 时
  else {
    if (probability > probThreshold) {
      // 条件4: 任一条件满足则为 fake
      if (is_AI == 1 ) {
        document.getElementById('My_advice').textContent ='此文本内容很可能为AI生成，请谨慎阅读';
        return 'fake';
      }else if(isObjectivityLow){
        document.getElementById('My_advice').textContent ='文本写作风格缺乏客观性，请谨慎阅读';
        return 'fake';
      }else {
        document.getElementById('My_advice').textContent ='此文本内容真实性较高';
        console.log("应用 real 样式");
        return 'real';
      }
    }
    else { // probability <= 0.65
       if (is_AI == 1 ) {
        document.getElementById('My_advice').textContent ='此文本内容很可能为AI生成，请谨慎阅读';
        return 'fake';
      }else if(isObjectivityLow){
        document.getElementById('My_advice').textContent ='文本写作风格缺乏客观性，请谨慎阅读';
        return 'fake';
      }else if(hasNoMediaSources)  {
        document.getElementById('My_advice').textContent ='文本内容无权威来源，请客观阅读';
        return 'consider';
      }else {
        document.getElementById('My_advice').textContent ='文本内容较真实，请客观阅读';
        return 'real';
      }
    }
  }
}