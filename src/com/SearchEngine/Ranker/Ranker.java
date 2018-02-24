package com.SearchEngine.Ranker;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;

public class Ranker {

    /* The number of pages in the graph*/
    private Integer pagesCount;

    /* The Graph adjacency list */
    private HashMap<Integer, ArrayList<Integer>> inList = new HashMap<Integer, ArrayList<Integer>>();

    /* The out degrees of each page */
    private ArrayList<Integer> outDegrees;

    /* The ranks of the pages */
    private ArrayList<Double> pagesRank;

    /* The dumping factor */
    private final Double alpha = 0.85;

    /* The maximum number iterations */
    private final Integer maxIterations = 100;

    /* Add an arc to the graph */
    private void addArc(int from, int to) {
        if (inList.containsKey(to))
            inList.get(to).add(from);
        outDegrees.set(from, outDegrees.get(from) + 1);
    }

    /* Initialize all vectors */
    private void initializeLists(int pagesCount) {
        this.pagesCount = pagesCount;

        outDegrees = new ArrayList<Integer>();
        pagesRank = new ArrayList<Double>();

        for (int i = 0; i < pagesCount; i++) {

            // Create a new list for each page
            ArrayList<Integer> newList = new ArrayList<Integer>();
            inList.put(i, newList);

            outDegrees.add(0);
            pagesRank.add(1.0 / pagesCount); // Initialize at first with 1/n prob
        }
    }

    public void readFile(String filePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String strLine;

            initializeLists(5000);

            // Read file line by line
            while ((strLine = br.readLine()) != null) {

                String[] strs = strLine.trim().split(" ");
                Integer u = Integer.parseInt(strs[0]);
                Integer v = Integer.parseInt(strs[1]);

                // Add arcs
                this.addArc(u, v);

                System.out.println(u.toString() + " " + v.toString());
            }
        } catch (Exception e) {
            //Catch exception if any
            System.err.println("Error: " + e.getMessage());
        }

        debugging();
    }

    public void rankPages() {

        for (int iteration = 0; iteration < maxIterations; iteration++) {

            Double danglingSum = 0.0, pagesRankSum = 0.0;

            for (int page = 0; page < pagesCount; page++) {
                if (outDegrees.get(page) == 0)
                    danglingSum += pagesRank.get(page);
                pagesRankSum += pagesRank.get(page);
            }

            // Normalize the PR(i) needed for the power method
            for (int page = 0; page < pagesCount; page++) {
                pagesRank.set(page, pagesRank.get(page) / pagesRankSum);
            }

            Double aPage = alpha * danglingSum / pagesRankSum * 1 / pagesCount; // Same for all pages
            Double boredProb = (1 - alpha) * (1 / pagesCount) * 1; // Same for all pages

            // Loop over all pages
            for (int page = 0; page < pagesCount; page++) {

                Double hPage = 0.0;

                if (inList.containsKey(page)) {
                    for (Integer from : inList.get(page)) {
                        hPage += (1.0 * pagesRank.get(from) / outDegrees.get(from));
                    }
                    hPage *= alpha; // Multiply by dumping factor.
                }

                pagesRank.set(page, (hPage + aPage + boredProb));
            }
        }
    }

    // Debugging @Samir55
    private void debugging() {
        System.out.println("The saved edges printing");
        System.out.println("===========================================");
        for (Integer vertex : inList.keySet()) {
            System.out.println("For the vertex " + vertex.toString());
            System.out.println("OutDegrees " + outDegrees.get(vertex).toString());
            System.out.println("Vertices pointing to this vertex ");
            if (inList.get(vertex).isEmpty()) {
                System.out.println("nil");
            } else {
                for (Integer to : inList.get(vertex)) {
                    System.out.print(to.toString() + " ");
                }
                System.out.println();
            }
            System.out.println("===========================================");
        }
    }

}
