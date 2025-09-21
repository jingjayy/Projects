/* import job dataset*/
PROC IMPORT datafile = '/home/u63869903/grp asg/job_cleanData.csv'
dbms = csv replace
out = WORK.JOB;
getnames = yes;
guessingrows = max;
RUN;

/* check the informat and format of the dataset*/
TITLE "Dataset Structure: Job Dataset";
PROC CONTENTS data=work.job; 
RUN;
TITLE;

DATA work.job ;
INFILE '/home/u63869903/grp asg/job_cleanData.csv' dlm=',' 
missover dsd lrecl=32767 firstobs=2 obs=1001;
 
informat job_ID best32. ;
informat job_title $31. ;
informat company_id best32. ;
informat comp_name $70. ;
informat work_type $7. ;
informat involvement $10. ;
informat employees_count best32. ;
informat total_applicants best32. ;
informat linkedin_followers best32. ;
informat job_details $12569. ;
informat details_id best32. ;
informat industry $51. ;
informat level $16. ;
informat City $26. ;
informat State $17. ;
format job_ID best12. ;
format job_title $31. ;
format company_id best12. ;
format comp_name $70. ;
format work_type $7. ;
format involvement $10. ;
format employees_count best12. ;
format total_applicants best12. ;
format linkedin_followers best12. ;
format job_details $12569. ;
format details_id best12. ;
format industry $51. ;
format level $16. ;
format City $26. ;
format State $17. ;

INPUT job_ID
	  job_title  $
	  company_id
	  comp_name  $
	  work_type  $
	  involvement  $
	  employees_count
	  total_applicants
	  linkedin_followers
	  job_details  $
	  details_id
	  industry  $
	  level  $
	  City  $
	  State  $;
           
/* set ERROR detection macro variable */
IF _ERROR_ THEN CALL symputx('_EFIERR_',1);  

/* check missing values of each variables */
	/* array for numeric variables */
    ARRAY num_vars {*} _NUMERIC_;
    /* array for character variables */
    ARRAY char_vars {*} _CHARACTER_;
    
    /* count missing values */
    missing_count = 0;
    
    /* loop through numeric variables */
    DO i = 1 TO DIM(num_vars);
        IF MISSING(num_vars{i}) THEN missing_count + 1;
    END;
    
    /* loop through character variables */
    DO i = 1 TO DIM(char_vars);
        IF MISSING(char_vars{i}) THEN missing_count + 1;
    END;
    
    /* replace 'Not Avilable' with '.' */
    DO i = 1 TO DIM(char_vars);
        IF char_vars{i} = 'Not Avilable' THEN char_vars{i} = '.'; 
    END;
  
/* filter out rows with 'Hybrid'*/
IF work_type = 'On-site' OR work_type = 'Remote';

KEEP job_title comp_name work_type total_applicants industry missing_count;
RUN;

/* data exploration on job dataset*/
TITLE "Descriptive Statistics for Total Number of Applicants for Job Dataset";
PROC MEANS DATA=work.job N MEAN STD MIN MAX MAXDEC=2;
    VAR total_applicants;
RUN;
TITLE;

TITLE "Frequency Distribution of Employment Type and Job Titles for Job Dataset";
PROC FREQ DATA=work.job;
    TABLES work_type job_title;
RUN;
TITLE;

TITLE "Total Number of Applicants by Work_Type for Job Dataset";
PROC SGPLOT DATA=work.job;
    VBAR work_type / RESPONSE=total_applicants STAT=SUM DATALABEL;
    YAXIS LABEL="Total Number of Applicants";
    XAXIS LABEL="Work Type";
RUN;
TITLE;

/* import productivity dataset*/
PROC IMPORT datafile = '/home/u63869903/grp asg/remote_work_productivity.csv'
dbms = csv replace
out = WORK.productivity;
getnames = yes;
guessingrows = max;
RUN;

/* check the informat and format of the dataset*/
TITLE "Dataset Structure: Productivity Dataset";
PROC CONTENTS data=work.productivity; 
RUN;
TITLE;

DATA work.productivity ;
INFILE '/home/u63869903/grp asg/remote_work_productivity.csv' dlm=',' 
missover dsd lrecl=32767 firstobs=2;

informat Employee_ID best32.;
informat Employment_Type $9.;
informat Hours_Worked_Per_Week best32.;
informat Productivity_Score best32.;
informat Well_Being_Score best32.;
format Employee_ID best12.;
format Employment_Type $9.;
format Hours_Worked_Per_Week best12.;
format Productivity_Score best12.;
format Well_Being_Score best12.;

INPUT Employee_ID
	  Employment_Type $
	  Hours_Worked_Per_Week
	  Productivity_Score
	  Well_Being_Score;

DROP Employee_ID;

/* set ERROR detection macro variable */
IF _ERROR_ THEN CALL symputx('_EFIERR_',1);  

/* replace 'In-Office' with 'On-site'*/
IF Employment_Type ='In-Office' 
	THEN Employment_Type = 'On-site'; 
RUN;

/* data exploration for productivity dataset*/
TITLE "Descriptive Statistics for Productivity and Well-Being Metrics for Productivity Dataset";
PROC MEANS DATA=work.productivity N MEAN STD MIN MAX MAXDEC=2;
    VAR Hours_Worked_Per_Week Productivity_Score Well_Being_Score;
RUN;
TITLE;

TITLE "Frequency Distribution of Employment Type for Productivity Dataset";
PROC FREQ DATA=work.productivity;
    TABLES Employment_Type;
RUN;
TITLE;

TITLE "Distribution of Employment Type in Productivity Dataset";
PROC SGPLOT DATA=work.productivity;
    VBAR Employment_Type / DATALABEL;
    XAXIS LABEL="Employment Type";
    YAXIS LABEL="Frequency";
RUN;
TITLE;

/* standardize variable length for*/
DATA work.job;
    LENGTH Employment_Type $9; 
    SET work.job (RENAME=(work_type=Employment_Type));
RUN;

DATA work.productivity;
    LENGTH Employment_Type $9;
    SET work.productivity;
RUN;

/* add row number to create an alignment because it doesnt work without row alignment*/
PROC SORT DATA=work.job;
    BY Employment_Type;
RUN;

DATA work.job_with_index;
    SET work.job;
    Row_Index = _N_;  
RUN;

PROC SORT DATA=work.productivity;
    BY Employment_Type;
RUN;

DATA work.productivity_with_index;
    SET work.productivity;
    Row_Index = _N_;  
RUN;

/* merge both datasets with row numbers*/
/* a. is job dataset, b. is productivity dataset*/
PROC SQL;
    CREATE TABLE work.merged_data AS
    SELECT 
        a.Employment_Type,
        a.Row_Index,
        a.job_title,
        a.comp_name,
        a.total_applicants,
        a.industry,
        b.Hours_Worked_Per_Week,
        b.Productivity_Score,
        b.Well_Being_Score
    FROM work.job_with_index AS a
    LEFT JOIN work.productivity_with_index AS b
    ON a.Employment_Type = b.Employment_Type
       AND a.Row_Index = b.Row_Index;
QUIT;

/* print the result of merged dataset (uncleaned)*/
TITLE "Merged Dataset (Uncleaned): On-Site and Remote Employment Analysis";
PROC PRINT DATA=work.merged_data;
    VAR Employment_Type job_title comp_name total_applicants industry Hours_Worked_Per_Week 
    Productivity_Score Well_Being_Score;
RUN;
TITLE;

/* explore merged dataset (uncleaned)*/
TITLE "Descriptive Statistics for Merged Dataset (Uncleaned)";
PROC MEANS DATA=work.merged_data N MEAN STD MIN MAX MAXDEC=2;
    VAR Hours_Worked_Per_Week Productivity_Score Well_Being_Score total_applicants;
RUN;

TITLE "Frequency Distribution of Employment Type and Industry in Merged Dataset (Uncleaned)";
PROC FREQ DATA=work.merged_data;
    TABLES Employment_Type job_title comp_name industry;
RUN;
TITLE;

/* there are blank values in some columns*/
/* remove rows with missing values in any column 
and add derived variable: Productivity_Per_Hour*/
DATA work.cleaned_data;
    SET work.merged_data;
    IF NMISS(Hours_Worked_Per_Week, Productivity_Score, Well_Being_Score) = 0 
    AND CMISS(Employment_Type, job_title, comp_name, industry) = 0;
    Productivity_Per_Hour = Productivity_Score / Hours_Worked_Per_Week;
    FORMAT Productivity_Per_Hour 8.2;
RUN;
/* log the number of rows removed */
PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :merged_row_count
    FROM work.merged_data;
QUIT;

PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :cleaned_row_count
    FROM work.cleaned_data;
QUIT;

%LET removed_rows = %EVAL(&merged_row_count - &cleaned_row_count);
%PUT NOTE: Number of rows removed = &removed_rows; /*31 rows removed*/

/* validation here is to compare metrics between merged (uncleaned) and merged (cleaned) datasets */
TITLE "Validation Check For Merged Dataset Metrics";
PROC MEANS DATA=work.merged_data N MEAN STD MIN MAX MAXDEC=2;
    VAR Hours_Worked_Per_Week Productivity_Score Well_Being_Score total_applicants;
RUN;
TITLE;

TITLE "Validation Check For Final Cleaned Dataset Metrics";
PROC MEANS DATA=work.cleaned_data N MEAN STD MIN MAX MAXDEC=2;
    VAR Hours_Worked_Per_Week Productivity_Score Well_Being_Score total_applicants;
RUN;
TITLE;

TITLE "Validation Check For Employment Type Distribution";
PROC FREQ DATA=work.merged_data;
    TABLES Employment_Type / MISSING;
RUN; /* no missing value therefore, no missing value column at the table*/

PROC FREQ DATA=work.cleaned_data;
    TABLES Employment_Type / MISSING;
RUN; /* no missing value therefore, no missing value column at the table*/
TITLE;

/* ensure no duplicate key on cleaned dataset*/
PROC SORT DATA=work.cleaned_data OUT=work.cleaned_data_no_dups NODUPKEY;
    BY Employment_Type job_title comp_name total_applicants industry Hours_Worked_Per_Week 
        Productivity_Score Well_Being_Score Productivity_Per_Hour;
RUN;

/* aggregate data to compute average metrics by Employment_Type */
PROC SQL;
    CREATE TABLE work.summary_by_employment AS
    SELECT 
        Employment_Type,
        COUNT(*) AS Count_Records,
        AVG(Productivity_Score) AS Avg_Productivity,
        AVG(Well_Being_Score) AS Avg_Well_Being,
        AVG(Hours_Worked_Per_Week) AS Avg_Hours
    FROM work.cleaned_data
    GROUP BY Employment_Type;
QUIT;

/* print the summary by Employment_Type table */
TITLE "Summary Statistics by Employment Type";
PROC PRINT DATA=work.summary_by_employment NOOBS;
    VAR Employment_Type Count_Records Avg_Productivity Avg_Well_Being Avg_Hours;
    FORMAT Avg_Productivity Avg_Well_Being Avg_Hours 8.2;
RUN;
TITLE;

TITLE "Average Productivity and Well-Being by Employment Type for Productivity Dataset";
PROC SGPLOT DATA=work.summary_by_employment;
    VBAR Employment_Type / RESPONSE=Avg_Productivity DATALABEL BARWIDTH=0.4 TRANSPARENCY=0.1
                           LEGENDLABEL="Average Productivity";
    VBAR Employment_Type / RESPONSE=Avg_Well_Being DATALABEL BARWIDTH=0.4 TRANSPARENCY=0.1
                           LEGENDLABEL="Average Well-Being";
    YAXIS LABEL="Average Score";
    XAXIS LABEL="Employment Type";
    KEYLEGEND / POSITION=BOTTOM LOCATION=OUTSIDE ACROSS=1;
RUN;
TITLE;

/* print final cleaned data with no duplicate key*/
TITLE "Final Cleaned Dataset: On-Site and Remote Employment Analysis";
PROC PRINT DATA=work.cleaned_data_no_dups;
    VAR Employment_Type job_title comp_name total_applicants industry Hours_Worked_Per_Week 
        Productivity_Score Well_Being_Score Productivity_Per_Hour;
RUN;
TITLE;

/* validation results are recorded to ensure no missing value*/
TITLE "Missing Value Check: Final Cleaned Dataset";
PROC FREQ DATA=work.cleaned_data_no_dups;
    TABLES Employment_Type job_title comp_name industry / MISSING;
RUN;
TITLE;

/*RESEARCH QUESTION 1: IS THERE A RELATIONSHIP 
  BETWEEN EMPLOYMENT TYPE AND WELL BEING SCORE?*/
DATA work.EmploymentType_WellBeingScore;
	SET work.cleaned_data_no_dups;
	LENGTH well_being_status $ 17;

/*if-else statement to see if employee well_being_scores are good or bad*/
	IF Well_Being_Score <= 25 THEN well_being_status = "Bad";
	ELSE IF Well_Being_Score >= 26 AND Well_Being_Score <= 50 THEN well_being_status = "Below_Average";
	ELSE IF Well_Being_Score >= 51 AND Well_Being_Score <= 75 THEN well_being_status = "Above_Average";
	ELSE IF Well_Being_Score >= 76 AND Well_Being_Score <= 100 THEN well_being_status = "Good";
	ELSE IF Well_Being_Score = . THEN well_being_status = .;
	ELSE well_being_status = "Insufficient_Data";
RUN;	

/*create a table to see how many falls onto each category in well_being_status*/
PROC FREQ DATA=WORK.EMPLOYMENTTYPE_WELLBEINGSCORE;
    TABLES well_being_status / NOCUM;
TITLE "Distribution of Employee Well-Being Status Based on Well-Being Scores";
RUN;

/*create a boxplot to visualize between well_being_score and employment_type*/
PROC SGPLOT DATA=work.EmploymentType_WellBeingScore;
    VBOX Well_Being_Score / CATEGORY=Employment_Type;
    TITLE "Box Plot of Well-Being Score by Employment Type";
RUN;

/*run a t-test for both variables in */
PROC TTEST DATA=WORK.EMPLOYMENTTYPE_WELLBEINGSCORE;
    CLASS Employment_Type;
    VAR Well_Being_Score;
    TITLE "T-Test for Well-Being Score by Employment Type";
RUN;

/*run chi square test to test if the variables in employment type is independent or associated*/
PROC FREQ DATA=WORK.EMPLOYMENTTYPE_WELLBEINGSCORE;
    TABLES Employment_Type * well_being_status / CHISQ EXACT;
    TITLE "Relationship Between Employment Type and Well-Being Status  (Fischer's Test)";
RUN;

/*RESEARCH QUESTION 2*/
PROC IMPORT DATAFILE='/home/u63869903/grp asg/remote_work_productivity.csv'
    OUT=WORK.REMOTE_WORK_PRODUCTIVITY
    DBMS=CSV
    REPLACE;
    GETNAMES=YES;
RUN;

/*Creates a Box Plot to visualise a relationship between employment_type and productivity_score*/
PROC SGPLOT DATA=remote_work_productivity;
    VBOX Productivity_Score / CATEGORY=Employment_Type;
    TITLE "Productivity Score by Employment Type";
RUN;

/*Creates a Bar Chart to visualise a relationship between employment_type and productivity_score*/
PROC MEANS DATA=remote_work_productivity NOPRINT;
    CLASS Employment_Type;
    VAR Productivity_Score;
    OUTPUT OUT=AverageScores MEAN=Avg_Productivity;
RUN;

PROC SGPLOT DATA=AverageScores;
    VBAR Employment_Type / RESPONSE=Avg_Productivity DATALABEL;
    TITLE "Average Productivity Score by Employment Type";
RUN;

/*Anova*/
PROC ANOVA DATA=remote_work_productivity;
    CLASS Employment_Type;
    MODEL Productivity_Score = Employment_Type;
    TITLE "ANOVA for Productivity Score by Employment Type";
RUN;

PROC ANOVA DATA=remote_work_productivity;
    CLASS Employment_Type;
    MODEL Well_Being_Score = Employment_Type;
    TITLE "ANOVA for Well-Being Score by Employment Type";
RUN;

PROC CORR DATA=remote_work_productivity;
    VAR Hours_Worked_Per_Week Productivity_Score Well_Being_Score;
    TITLE "Correlation Between Work Hours and Scores";
RUN;

/*RESEARCH QUESTION 4*/
/* Create a new variable for Productivity Per Hour */
DATA work.HoursWorked_Productivity;
    SET work.cleaned_data_no_dups;
    LENGTH Productivity_Per_Hour 8.; /* Define the new variable */
    IF Hours_Worked_Per_Week > 0 THEN Productivity_Per_Hour = Productivity_Score / Hours_Worked_Per_Week;
    ELSE Productivity_Per_Hour = .; /* Handle cases where hours worked is zero */
RUN;

/* Scatter plot with trend line */
PROC SGPLOT DATA=work.HoursWorked_Productivity;
    SCATTER X=Hours_Worked_Per_Week Y=Productivity_Per_Hour / MARKERATTRS=(SYMBOL=circlefilled COLOR=blue SIZE=10);
    REG X=Hours_Worked_Per_Week Y=Productivity_Per_Hour / LINEATTRS=(COLOR=red THICKNESS=2);
    TITLE "Scatter Plot with Trend Line: Hours Worked vs. Productivity Per Hour";
    XAXIS LABEL="Hours Worked Per Week";
    YAXIS LABEL="Productivity Per Hour";
RUN;

/* Correlation analysis */
PROC CORR DATA=work.HoursWorked_Productivity;
    VAR Hours_Worked_Per_Week Productivity_Per_Hour;
    TITLE "Correlation Analysis: Hours Worked and Productivity Per Hour";
RUN;

/* Linear regression analysis */
PROC REG DATA=work.HoursWorked_Productivity;
    MODEL Productivity_Per_Hour = Hours_Worked_Per_Week;
    TITLE "Linear Regression: Productivity Per Hour vs. Hours Worked";
RUN;
QUIT;

/* Categorize Hours Worked into Groups for Comparison */
PROC FORMAT;
    VALUE hours_grp
        LOW-30 = 'Short'
        31-40 = 'Medium'
        41-HIGH = 'Long';
RUN;

DATA work.HoursWorked_Productivity;
    SET work.HoursWorked_Productivity;
    LENGTH Hours_Category $ 10; /* Define the new category variable */
    Hours_Category = PUT(Hours_Worked_Per_Week, hours_grp.);
RUN;

/* Box plot for Productivity Per Hour by Hours Worked Category */
PROC SGPLOT DATA=work.HoursWorked_Productivity;
    VBOX Productivity_Per_Hour / CATEGORY=Hours_Category;
    TITLE "Box Plot: Productivity Per Hour by Hours Worked Category";
    XAXIS LABEL="Hours Worked Category";
    YAXIS LABEL="Productivity Per Hour";
RUN;

/* ANOVA Test for Differences in Productivity Per Hour by Hours Worked Category */
PROC ANOVA DATA=work.HoursWorked_Productivity;
    CLASS Hours_Category;
    MODEL Productivity_Per_Hour = Hours_Category;
    MEANS Hours_Category / TUKEY;
    TITLE "ANOVA Test: Productivity Per Hour by Hours Worked Category";
RUN;
QUIT;