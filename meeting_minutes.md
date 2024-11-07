# 28 Oct 2024
- stick with chest X-rays?
- IID vs Non-IID
    - how can we divide them into Non-IID
    - what kind of work would we need to do to make it IID? (Kim)
    - Number of infections (sample size), if they have an infection or not, severity of the infection (percentage of the total lung region)
        - basic algorithm steps
        - put this in the context of the initial infection
        - EDA (exploratory data analysis), make charts detailing counts of severities of lung counts to justify how we distribute the data.
- will use UNet or another similar CNN architecture

# 4 Nov 2024
- David has made initial model with UNet architecture
- accuracy is not that great, biases infections with two lungs
- Meeting 1:30 - 4:00